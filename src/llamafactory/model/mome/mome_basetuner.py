import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict

from .mome_index import LaminiIndex
from lamini_ml.mome.model_definition.constants import get_hidden_size
from peft.tuners.tuners_utils import BaseTunerLayer


class MoMEAttentionAdaptor(nn.Module, BaseTunerLayer):
    """
    MoMEAttentionAdaptor integrates the base attention layer with soft attention over a trainable index.
    This enables dynamic key/value retrieval and backpropagation into the index embeddings.
    """

    adapter_layer_names = ("query_in", "query_out", "value_in", "value_out", "index")
    other_param_names = ("r", "scaling")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        index_k: int = 5,
        index: Optional[LaminiIndex] = None,
        **kwargs,
    ):
        super().__init__()
        BaseTunerLayer.__init__(self, base_layer)

        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.index_k = index_k
        self.index = nn.ModuleDict({})

        # LoRA-style projections
        self.query_in = nn.ModuleDict({})
        self.query_out = nn.ModuleDict({})
        self.value_in = nn.ModuleDict({})
        self.value_out = nn.ModuleDict({})
        self.lora_dropout = nn.ModuleDict({})

        # Initialize adapter
        if r > 0:
            self._active_adapter = adapter_name
            self.update_layer(adapter_name, r, index, index_k)
        else:
            self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        index: LaminiIndex,
        index_k: int,
    ) -> None:
        """
        Dynamically creates LoRA projections and stores a LaminiIndex for the adapter.

        Args:
            adapter_name: Name of the adapter.
            r: LoRA rank.
            index: LaminiIndex instance with trainable `keys` and `values`.
            index_k: Number of nearest neighbors to retrieve (used for attention windowing).
        """
        hidden_size = get_hidden_size(self.base_layer)
        index_dimension = index.embedding_dim

        # Query projections
        self.query_in[adapter_name] = nn.Linear(hidden_size, r, bias=False)
        self.query_out[adapter_name] = nn.Linear(r, index_dimension, bias=False)

        # Value projections
        self.value_in[adapter_name] = nn.Linear(index_dimension, r, bias=False)
        self.value_out[adapter_name] = nn.Linear(r, hidden_size, bias=False)

        # Store LaminiIndex as an adapter layer
        self.index[adapter_name] = index

        # Initialize weights
        nn.init.kaiming_uniform_(self.query_in[adapter_name].weight, a=torch.sqrt(torch.tensor(5)))
        nn.init.zeros_(self.query_out[adapter_name].weight)

        nn.init.kaiming_uniform_(self.value_in[adapter_name].weight, a=torch.sqrt(torch.tensor(5)))
        nn.init.zeros_(self.value_out[adapter_name].weight)

        # Scaling by rank
        self.scaling[adapter_name] = r
        self.r[adapter_name] = r

        # Register adapter as active
        self.set_adapter([adapter_name])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass combining base attention with index-based attention.
        """
        # Base layer forward
        base_output = self.base_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **kwargs
        )

        base_output_tensor = base_output[0] if isinstance(base_output, tuple) else base_output
        active_adapter = self.active_adapter[0] if isinstance(self.active_adapter, list) else self.active_adapter

        if not active_adapter or active_adapter not in self.query_in:
            return base_output

        # Projection via LoRA
        query = self.query_out[active_adapter](
            self.query_in[active_adapter](hidden_states)
        )

        # Retrieve key/value from LaminiIndex
        key, value = self._get_key_value_from_index(query, active_adapter)

        # Compute attention with retrieved key/value
        mome_attention = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            is_causal=True
        )

        # Final value projection
        mome_value_proj = self.value_out[active_adapter](
            self.value_in[active_adapter](mome_attention)
        ) * self.scaling[active_adapter]

        # Combine base and MoME attention outputs
        combined_output = base_output_tensor + mome_value_proj

        return (combined_output,) + base_output[1:] if isinstance(base_output, tuple) else combined_output

    def _get_key_value_from_index(self, query: Tensor, adapter_name: str) -> Tuple[Tensor, Tensor]:
        """
        Retrieve top-k key/value from the index using soft dot product attention.
        """
        b, s, d = query.shape
        index = self.index[adapter_name]

        # Reshape query to (b * s, d)
        query_flat = query.view(b * s, -1)

        # Compute similarity with all keys in the index
        with torch.no_grad():
            attn_weights = torch.matmul(query_flat, index.keys.t())
            _, top_indices = torch.topk(attn_weights, k=self.index_k, dim=-1)

        # Flatten indices for gather
        flat_indices = top_indices.view(-1)

        # Gather top-k key/value embeddings
        key = index.keys[flat_indices].view(b, self.index_k * s, -1)
        value = index.values[flat_indices].view(b, self.index_k * s, -1)

        return key, value