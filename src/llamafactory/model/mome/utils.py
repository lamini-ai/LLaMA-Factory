import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from .mome_index import LaminiIndex 
from llamafactory.data.data_utils import DatasetModule



def infer_attention_embed_dim(attn_mod: nn.Module) -> int:
    """
    Infer the model (token) embedding dimension from a HF attention module.
    Works for Llama, DeepSeek, Qwen, Gemma, vanilla nn.MultiheadAttention, etc.
    """
    # 1) PyTorch nn.MultiheadAttention & some HF ports
    if hasattr(attn_mod, "embed_dim"):
        return int(attn_mod.embed_dim)

    # 2) Common across modern HF LLMs (q_proj is an nn.Linear)
    if hasattr(attn_mod, "q_proj"):
        return int(attn_mod.q_proj.in_features)

    # 3) Some older/bert‑like blocks use 'query'
    if hasattr(attn_mod, "query"):
        return int(attn_mod.query.in_features)

    # 4) Out‑projection is a safe last resort
    if hasattr(attn_mod, "out_proj"):
        return int(attn_mod.out_proj.in_features)

    raise AttributeError(
        f"Cannot locate embed dimension on {attn_mod.__class__.__name__}"
    )


def find_and_initialize_mome_adapters(        
        model: nn.Module, 
        dataset_module: DatasetModule, 
        finetuning_args: "FinetuningArguments",
    ) -> None:
    """
    Find and initialize MoME adapters in the model.
    """
    
    from .mome_basetuner import MoMEAttentionAdaptor
    
    # ---- 1. Precompute the index ----
    dataset = dataset_module["train_dataset"]
    shared_index = LaminiIndex()
    shared_index.initialize(
        dataset=iter(dataset),
        index_k=finetuning_args.index_k,
        sentence_transformer_name=finetuning_args.sentence_transformer_name,
        sentence_transformer_dim=finetuning_args.sentence_transformer_dim,
        cache_dir=finetuning_args.sentence_transformer_cache_dir,
        sentence_transformer_batch_size=finetuning_args.sentence_transformer_batch_size,
    )

    # ---- 2. Find all adaptors ----
    adapters: list[tuple[str, MoMEAttentionAdaptor]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoMEAttentionAdaptor)
    ]
    print(f"Found {len(adapters)} MoMEAttentionAdaptor layers")

    # ---- 3. Parallel update config + bind shared index ----
    def _init_adapter(item):
        name, adaptor = item
        adaptor.update_layer(index_dimension=finetuning_args.sentence_transformer_dim)
        adaptor.initialize_index_from_prebuilt_index(shared_index) 
        return name

    with ThreadPoolExecutor() as pool:
        for layer_name in pool.map(_init_adapter, adapters):
            print(f"  ↳ adaptor '{layer_name}' initialized")
