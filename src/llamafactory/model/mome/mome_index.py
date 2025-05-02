from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterator
import logging
from itertools import islice

from .constants import EMBEDDING_MODEL_NAME, SENTENCE_TRANSFORMER_DIM

logger = logging.getLogger(__name__)

class LaminiIndex(nn.Module):
    """
    A differentiable index that uses soft attention over key/value embeddings.
    Embeddings are initialized using the input dataset.
    Dataset is processed in batches to prevent GPU OOM.
    The number of embeddings is determined by the length of the dataset.
    """

    def __init__(
        self,
        dataset: Iterator[str],
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize a differentiable key/value memory bank using the given dataset.

        Args:
            dataset (Iterator[str]): Iterable of text examples (assumed to be finite).
            embedding_model_name (str): Name of the Sentence Transformer model.
            cache_dir (Optional[str]): Directory to cache the embedding model.
            batch_size (int): Batch size for GPU memory efficiency and OOM prevention.
        """
        super().__init__()

        # Load the sentence transformer model
        self.embedding_model = SentenceTransformer(embedding_model_name, cache_folder=cache_dir)
        self.embedding_dimension = min(
            self.embedding_model.get_sentence_embedding_dimension(),
            SENTENCE_TRANSFORMER_DIM,
        )

        # Generate embeddings from dataset
        embeddings = self._generate_embeddings(dataset, batch_size)
        self.num_embeddings = embeddings.size(0)

        # Initialize keys and values using these embeddings
        self.keys = nn.Parameter(embeddings.clone())
        self.values = nn.Parameter(embeddings.clone())

        logger.info(f"Initialized LaminiIndex with {self.num_embeddings} embeddings of dimension {self.embedding_dimension}")

    def _generate_embeddings(self, dataset: Iterator[str], batch_size: int) -> torch.Tensor:
        """
        Generate embeddings from the dataset in batches.

        Args:
            dataset (Iterator[str]): Iterable of text examples.
            batch_size (int): Batch size for processing.

        Returns:
            torch.Tensor: Concatenated embeddings tensor.
        """
        texts = []
        batch_count = 0
        embeddings = []

        with torch.no_grad():
            while True:
                try:
                    # Get a batch of items using islice
                    batch = list(islice(dataset, batch_size))
                    if not batch:
                        break

                    logger.info(f"Encoding batch {batch_count + 1} of approx {len(batch)} texts...")

                    # Encode batch and convert to tensor
                    batch_embeddings = self.embedding_model.encode(
                        sentences=batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )

                    if isinstance(batch_embeddings, torch.Tensor):
                        embeddings.append(batch_embeddings)
                    else:
                        # Fallback if encode returns a numpy array or other
                        embeddings.append(torch.tensor(batch_embeddings))

                    texts.extend(batch)
                    batch_count += 1

                except StopIteration:
                    break

        if not texts:
            raise ValueError("The provided dataset is empty — cannot initialize embeddings.")

        # Concatenate all batch embeddings into a single tensor
        return torch.cat(embeddings, dim=0)

    
    def forward(
        self,
        query: torch.Tensor,
        k: int = 8,
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
        topk_vals, topk_idx = probs.topk(k, dim=-1)        # (B*L, k)
        
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