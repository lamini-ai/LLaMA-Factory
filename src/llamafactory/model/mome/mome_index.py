from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple, Iterator
import logging
from itertools import islice

logger = logging.getLogger(__name__)

class LaminiIndex(nn.Module):
    def __init__(self):
        """
        Initialize an empty LaminiIndex. Must call `.initialize()` before use.
        """
        super().__init__()
        self.embedding_model = None
        self.embedding_dimension = None
        self.num_embeddings = None
        self.keys = None
        self.values = None

    def initialize(
        self,
        dataset: Iterator[str],
        index_k: int,
        sentence_transformer_name: str,
        sentence_transformer_dim: str,
        cache_dir: Optional[str] = None,
        sentence_transformer_batch_size: int = 32,
    ):
        """
        Initialize the embedding model and key/value memory with dataset.

        Args:
            dataset (Iterator[str]): Iterable of text examples.
            sentence_transformer_name (str): Name of the Sentence Transformer model.
            sentence_transformer_dim (str): Dimension of the Sentence Transformer model.
            cache_dir (Optional[str]): Directory to cache the embedding model.
            sentence_transformer_batch_size (int): Batch size for GPU memory efficiency and OOM prevention.
        """
        self.index_k = index_k
        logger.info(f"Loading embedding model '{sentence_transformer_name}' from cache_dir={cache_dir}")
        self.embedding_model = SentenceTransformer(sentence_transformer_name, cache_folder=cache_dir)

        self.embedding_dimension = min(
            self.embedding_model.get_sentence_embedding_dimension(),
            sentence_transformer_dim,
        )
        logger.info(f"Embedding model loaded with dimension {self.embedding_dimension}")

        embeddings = self._generate_embeddings(dataset, sentence_transformer_batch_size)
        self.num_embeddings = embeddings.size(0)

        self.keys = nn.Parameter(embeddings.clone())
        self.values = nn.Parameter(embeddings.clone())

        logger.info(f"Initialized LaminiIndex with {self.num_embeddings} embeddings of dimension {self.embedding_dimension}")

    def _generate_embeddings(
        self,
        dataset: Iterable[str],          # <- works for list OR iterator
        batch_size: int
    ) -> torch.Tensor:

        dataset_iter = iter(dataset)     # <‑‑ key change: one iterator
        embeddings   = []
        batch_count  = 0

        with torch.no_grad():
            while True:
                batch = list(islice(dataset_iter, batch_size))
                if not batch:            # dataset exhausted – we’re done
                    break

                logger.info(
                    f"Encoding batch {batch_count + 1} (size={len(batch)})…"
                )

                batch_emb = self.embedding_model.encode(
                    sentences=batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                embeddings.append(
                    batch_emb
                    if isinstance(batch_emb, torch.Tensor)
                    else torch.tensor(batch_emb)
                )
                batch_count += 1

        if not embeddings:
            raise ValueError(
                "The provided dataset is empty — cannot initialize embeddings."
            )

        return torch.cat(embeddings, dim=0)

    
    def forward(
        self,
        query: torch.Tensor,
        tau: float = 0.5,
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
        probs = F.gumbel_softmax(
            logits, tau=tau, hard=False, dim=-1
        )                                            # (B*L, N)

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