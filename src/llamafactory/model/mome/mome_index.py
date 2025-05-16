import threading
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple
import logging
from itertools import islice

logger = logging.getLogger(__name__)

class LaminiIndex(nn.Module):


    _shared_keys: Optional[nn.Parameter] = None     # the one true key matrix
    _shared_dim: Optional[int] = None              # D, used for sanity checks
    _init_lock = threading.Lock()                  # protects first-time creation
    
    def __init__(self):
        """
        Initialize an empty LaminiIndex. Must call `.initialize()` before use.
        """
        super().__init__()
        self.embedding_dimension: Optional[int] = None
        self.num_embeddings: Optional[int] = None
        # keys will be registered later
        self.values: Optional[nn.Parameter] = None
        self.keys: Optional[nn.Parameter] = None
        self.index_k: Optional[int] = None
        
        
    def initialize_empty_weights(
        self,
        index_k: int,
        sentence_transformer_dim: int,
        num_experts: int,
    ) -> None:
        """
        Initialize the key matrix and create a trainable value table.
        """
        self.index_k = index_k
        self.keys = nn.Parameter(torch.zeros(num_experts, sentence_transformer_dim), requires_grad=False)
        self.values = nn.Parameter(torch.zeros(num_experts, sentence_transformer_dim), requires_grad=True)
        
    def initialize(
        self,
        dataset: Iterable[str],
        index_k: int,
        sentence_transformer_name: str,
        sentence_transformer_dim: int,
        cache_dir: Optional[str] = None,
        sentence_transformer_batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Build (or re-use) the key matrix and create a trainable value table.

        Args
        ----
        dataset : iterable of strings        – used only the first time
        index_k : int                        – top-k neighbours at query time
        sentence_transformer_name : str      – HF model id
        sentence_transformer_dim : int       – cap key dimension (≤ model dim)
        cache_dir : str or None              – local cache for HF models
        sentence_transformer_batch_size : int
        device : torch.device or None        – defaults to current CUDA or CPU
        """
        self.index_k = index_k
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # 1. Build shared keys once, protected by a lock
        # ------------------------------------------------------------------ #
        with LaminiIndex._init_lock:
            if LaminiIndex._shared_keys is None:
                logger.info("Building shared LaminiIndex keys …")

                st_model = SentenceTransformer(sentence_transformer_name,
                                               cache_folder=cache_dir)
                model_dim = st_model.get_sentence_embedding_dimension()
                self.embedding_dimension = min(model_dim, sentence_transformer_dim)

                # encode full dataset → tensor [N, D]
                embeddings = self._generate_embeddings(
                    st_model, dataset, sentence_transformer_batch_size
                )[:, : self.embedding_dimension]            # trim if needed
                LaminiIndex._shared_keys = nn.Parameter(
                    embeddings.to(device), requires_grad=False
                )
                LaminiIndex._shared_dim = self.embedding_dimension
                self.num_embeddings = embeddings.size(0)

                logger.info(
                    f"Shared keys created: {self.num_embeddings} × "
                    f"{self.embedding_dimension} on {device}"
                )
            else:
                # sanity-check dimension consistency
                if sentence_transformer_dim != LaminiIndex._shared_dim:
                    raise ValueError(
                        "LaminiIndex: attempted to create a second key matrix "
                        f"with dim={sentence_transformer_dim}, "
                        f"but existing shared dim={LaminiIndex._shared_dim}."
                    )
                self.embedding_dimension = LaminiIndex._shared_dim
                self.num_embeddings = LaminiIndex._shared_keys.size(0)
                logger.info("↪  Re-using cached LaminiIndex keys")

        # ------------------------------------------------------------------ #
        # 2.  Register the (shared) keys in *this* sub-module
        # ------------------------------------------------------------------ #
        # If you *never* want keys to appear in the optimizer, use register_buffer.
        
        self.keys = nn.Parameter(LaminiIndex._shared_keys.data, requires_grad=False)

        # ------------------------------------------------------------------ #
        # 3.  Each instance gets its own trainable value matrix
        # ------------------------------------------------------------------ #
        self.values = nn.Parameter(
            LaminiIndex._shared_keys.data.clone(), requires_grad=True
        )
        logger.info(
            f"Initialized LaminiIndex instance — values trainable, "
            f"keys shared (id={id(LaminiIndex._shared_keys)})"
        )

    def clone_with_shared_keys(self) -> "LaminiIndex":
        """
        Return a new LaminiIndex that *reuses* the class-level _shared_keys
        tensor but clones its trainable `values`.
        """
        new = LaminiIndex()
        # metadata
        new.embedding_dimension = self.embedding_dimension
        new.num_embeddings      = self.num_embeddings
        new.index_k             = self.index_k

        # tie the frozen key parameter
        new.keys = nn.Parameter(LaminiIndex._shared_keys.data, requires_grad=False)

        # give it its own value table
        new.values = nn.Parameter(self.values.data.clone(), requires_grad=True)
        return new

    def _generate_embeddings(
        self,
        st_model: SentenceTransformer,
        dataset: Iterable[str],
        batch_size: int
    ) -> torch.Tensor:

        dataset_iter = iter(dataset)
        embeddings   = []
        batch_count  = 0

        with torch.no_grad():
            while True:
                chunk = list(islice(dataset_iter, batch_size))
                if not chunk:
                    break

                logger.info(
                    f"Encoding batch {batch_count + 1} (size={len(chunk)})…"
                )

                batch_emb = st_model.encode(
                    sentences=chunk,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                embeddings.append(batch_emb)
                batch_count += 1

        if not embeddings:
            raise ValueError(
                "The provided dataset is empty — cannot initialize embeddings."
            )

        return torch.cat(embeddings, dim=0)

    
    def forward(
        self,
        query: torch.Tensor,
        tau: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gumbel-Softmax + hard Top-k mask differentiable retrieval

        Args:
            query (Tensor): (batch_size, seq_len, D)
            k (int): Number of memory entries to select per query
            tau (float): Gumbel-Softmax temperature
        Returns:
            key_vectors, value_vectors: (batch, seq_len, D)
        """
        # if torch.isnan(query).any():
        #     raise ValueError("Input query contains NaN values")
        
        B, L, D = query.shape
        q = query.view(B * L, D)                 # (B*L, D)

        # 1) Compute similarity logits
        logits = torch.matmul(q, self.keys.t())  # (B*L, N)

        # 2) Soft probs with Gumbel noise and scale by temperature
        #    -------------------------------------------------
        #    • Gumbel noise turns the arg-max selection problem
        #      into a *reparameterised* continuous sample,
        #      so gradients can flow through the sampling step
        #      (Gumbel-Softmax / Concrete distribution trick).
        #
        #    • The temperature τ controls sharpness:
        #        τ → ∞  ⇒ almost uniform (soft exploration)
        #        τ → 0  ⇒ approaches one-hot arg-max (hard selection)
        #      You can anneal τ during training to gradually
        #      move from soft to hard Top-k.
        #
        #    • Without the noise the model would always
        #      choose the same few keys early on, starving the
        #      rest of the table of gradient updates.
        
        # TODO: NaN issue: https://github.com/pytorch/pytorch/issues/22442
        # probs = F.gumbel_softmax(
        #     logits, tau=tau, hard=False, dim=-1
        # )                                            # (B*L, N)
        
        probs = F.softmax(logits / tau, dim=-1)

        # 4) Take the top k indices, make a "hard" one-hot mask
        topk_vals, topk_idx = probs.topk(self.index_k, dim=-1)        # (B*L, k)
        
        # (B*L, N) multiple one-hot vector 
        hard_mask = torch.zeros_like(probs).scatter_(
            -1, topk_idx, 1.0
        )
        
        # 5) Renormalise the hard mask so each row sums to 1
        #    ------------------------------------------------
        #    hard_mask is one-hot (k ones, others zero).  Dividing
        #    by its row-sum k turns those ones into 1/k, giving a
        #    *proper* probability distribution.  This keeps the
        #    later weighted sum (attn @ keys) a true convex average
        #    instead of scaling vectors by k.
        hard_mask = hard_mask / hard_mask.sum(-1, keepdim=True)  # (B*L, N)

        # 6) Straight-Through estimator: forward = hard, backward = soft
        #    -----------------------------------------------------------
        #    (hard_mask - probs).detach()   --> value is (hard - soft) but
        #                                       *no gradient* flows through it.
        #    + probs                        --> adds back the soft version
        #
        #    Result:
        #      * Forward pass  : attn == hard_mask  (strict Top-k)
        #      * Backward pass : ∇attn == ∇probs    (smooth gradients)
        #
        #    So we enjoy the efficiency/interpretability of a hard Top-k
        #    selection while still training with low-variance gradients.t
        attn = (hard_mask - probs).detach() + probs        # (B*L, N)

        # 7) Get weighted key / value
        key_vec  = torch.matmul(attn, self.keys)           # (B*L, D)
        value_vec = torch.matmul(attn, self.values)        # (B*L, D)

        # 8) Reshape to (batch, seq, embedding_dim)
        key_vec   = key_vec.view(B, L, D)
        value_vec = value_vec.view(B, L, D)
        return key_vec, value_vec



    def forward_alternative(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft attention over keys to compute a differentiable retrieval of values.

        Args:
            query (torch.Tensor): Shape (batch_size, seq_len, embedding_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - key_vectors: attention-weighted keys
                - value_vectors: attention-weighted values
        """
        batch_size, seq_len, _ = query.shape
        query_flat = query.view(batch_size * seq_len, -1)

        # Compute similarity (dot product) between query and keys
        # (batch*seq, num_embeddings)
        attn_weights = torch.matmul(query_flat, self.keys.t())
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute attention-weighted keys and values
        key_vectors = torch.matmul(attn_weights, self.keys)  # (batch*seq, embedding_dim)
        value_vectors = torch.matmul(attn_weights, self.values)  # (batch*seq, embedding_dim)

        # Reshape to (batch, seq, embedding_dim)
        key_vectors = key_vectors.view(batch_size, seq_len, -1)
        value_vectors = value_vectors.view(batch_size, seq_len, -1)

        return key_vectors, value_vectors