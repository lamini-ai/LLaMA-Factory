import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LaminiIndex(nn.Module):
    """
    A differentiable version of LaminiIndex using soft attention over key/value embeddings.
    Gradients flow into `keys` and `values` during training.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize a differentiable key/value memory bank.

        Args:
            num_embeddings (int): Number of key/value pairs in the memory.
            embedding_dim (int): Dimension of each embedding.
            init_embeddings (Optional[torch.Tensor]): Optional initial values for keys and values.
        """
        super().__init__()
        if init_embeddings is not None:
            assert init_embeddings.shape == (num_embeddings, embedding_dim), "Embedding shape mismatch."
            self.keys = nn.Parameter(init_embeddings.clone())
            self.values = nn.Parameter(init_embeddings.clone())
        else:
            self.keys = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
            self.values = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    # TODO: Verify this
    def soft_topk(self, query, keys, k):
        logits = torch.matmul(query, keys.t())
        gumbel_logits = F.gumbel_softmax(logits, tau=0.5, hard=False)
        top_k_weights, top_k_indices = torch.topk(gumbel_logits, k=k, dim=-1)

        # Use top_k_weights as soft weights
        key_vectors = torch.matmul(top_k_weights, keys)
        value_vectors = torch.matmul(top_k_weights, values)

        return key_vectors, value_vectors

    def update_embeddings(self, indices: torch.Tensor, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Update specified key/value embeddings using backpropagation.

        Args:
            indices (torch.Tensor): Indices to update (shape: [N])
            new_keys (torch.Tensor): New key embeddings (shape: [N, embedding_dim])
            new_values (torch.Tensor): New value embeddings (shape: [N, embedding_dim])
        """
        with torch.no_grad():
            self.keys[indices] = new_keys
            self.values[indices] = new_values

