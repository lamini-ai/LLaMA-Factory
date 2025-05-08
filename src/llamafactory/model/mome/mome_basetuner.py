import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Any, Union
from datasets import Dataset, DatasetDict, IterableDataset

from .mome_index import LaminiIndex
from .utils import infer_attention_embed_dim
from .constants import SENTENCE_TRANSFORMER_NAME, SENTENCE_TRANSFORMER_DIM
from peft.tuners.tuners_utils import BaseTunerLayer
# from peft.tuners.lora.layer import Linear, MultiheadAttention


class MoMEAttentionAdaptor(nn.Module, BaseTunerLayer):
    """
    MoMEAttentionAdaptor integrates the base attention layer with soft attention over a trainable index.
    This enables dynamic key/value retrieval and backpropagation into the index embeddings.
    """

    adapter_layer_names = ("lora_attn_query_in", "lora_attn_query_out", "lora_attn_value_in", "lora_attn_value_out", "lora_attn_index")
    other_param_names = ("r", "lora_attn_scaling")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ):
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer
        self.r = {adapter_name: r}
        self.lora_attn_scaling = {adapter_name: lora_alpha / r}
        self.lora_attn_index = nn.ModuleDict({})

        # LoRA-style projections
        self.lora_attn_query_in = nn.ModuleDict({})
        self.lora_attn_query_out = nn.ModuleDict({})
        self.lora_attn_value_in = nn.ModuleDict({})
        self.lora_attn_value_out = nn.ModuleDict({})
        self.lora_attn_dropout = nn.ModuleDict({})

        # Initialize adapter
        self._active_adapter = adapter_name

    def update_layer(
        self,
        index_dimension: int,
    ) -> None:
        """
        Dynamically creates LoRA projections and stores a LaminiIndex for the adapter.

        Args:
            adapter_name: Name of the adapter.
            r: LoRA rank.
            index: LaminiIndex instance with trainable `keys` and `values`.
        """
        hidden_size = infer_attention_embed_dim(self.base_layer)
        adapter_name = self._active_adapter

        # Query projections
        self.lora_attn_query_in[adapter_name] = nn.Linear(hidden_size, self.r[adapter_name], bias=False)
        self.lora_attn_query_out[adapter_name] = nn.Linear(self.r[adapter_name], index_dimension, bias=False)

        # Value projections
        self.lora_attn_value_in[adapter_name] = nn.Linear(index_dimension, self.r[adapter_name], bias=False)
        self.lora_attn_value_out[adapter_name] = nn.Linear(self.r[adapter_name], hidden_size, bias=False)

        # Placeholder for index, will call initialize_index_from_prebuilt_index() later
        self.lora_attn_index[adapter_name] = None

        # TODO: Explore more initialization methods
        nn.init.kaiming_uniform_(self.lora_attn_query_in[adapter_name].weight)
        nn.init.zeros_(self.lora_attn_query_out[adapter_name].weight)

        nn.init.kaiming_uniform_(self.lora_attn_value_in[adapter_name].weight)
        nn.init.zeros_(self.lora_attn_value_out[adapter_name].weight)


    def initialize_index_from_prebuilt_index(self, index: LaminiIndex):
        adapter_name = self._active_adapter[0] if isinstance(self._active_adapter, list) else self._active_adapter
        self.lora_attn_index[adapter_name] = index.clone_with_shared_keys()
        
        # Register adapter as active
        self.set_adapter([adapter_name])

    def initialize_index(self, 
                         dataset: Union[Dataset, DatasetDict, IterableDataset],
                         index_k: int,
                         sentence_transformer_name: str = SENTENCE_TRANSFORMER_NAME,
                         sentence_transformer_dim: str = SENTENCE_TRANSFORMER_DIM,
                         cache_dir: Optional[str] = None,
                         sentence_transformer_batch_size: int = 32):
        adapter_name = self._active_adapter[0] if isinstance(self._active_adapter, list) else self._active_adapter
        self.lora_attn_index[adapter_name].initialize(
            dataset, index_k, sentence_transformer_name, sentence_transformer_dim, cache_dir, sentence_transformer_batch_size)
        
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

        if not active_adapter or active_adapter not in self.lora_attn_query_in:
            return base_output

        # Projection via LoRA
        query = self.lora_attn_query_out[active_adapter](
            self.lora_attn_query_in[active_adapter](hidden_states)
        ) * self.lora_attn_scaling[active_adapter]

        # Retrieve key/value from LaminiIndex
        key, value = self.lora_attn_index[active_adapter](query)

        # Compute attention with retrieved key/value
        # pylint: disable=not-callable
        mome_attention = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            # attn_mask=attention_mask,
            dropout_p=0.1,
            is_causal=True,
            scale=None
        )

        # Final value projection
        mome_value_proj = self.lora_attn_value_out[active_adapter](
            self.lora_attn_value_in[active_adapter](mome_attention)
        ) * self.lora_attn_scaling[active_adapter]

        # Combine base and MoME attention outputs
        combined_output = base_output_tensor + mome_value_proj

        return (combined_output,) + base_output[1:] if isinstance(base_output, tuple) else combined_output

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "momeattn." + rep