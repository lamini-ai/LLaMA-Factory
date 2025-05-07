from typing import Union
import torch.nn as nn
from datasets import Dataset, DatasetDict, IterableDataset

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


def find_and_initialize_mome_adapters_old(
        model: nn.Module, 
        dataset_module: DatasetModule, 
        finetuning_args: "FinetuningArguments",
    ) -> None:
    """
    Recursively searches through a model's modules to locate all instances of MoMEAttentionAdaptor.
    For each found instance, it updates the layer configuration and initializes the index.

    Args:
        model (nn.Module): The neural network model to be searched.
        index_dimension (int): The dimension of the index used in the MoMEAttentionAdaptor.
        dataset (Union[Dataset, DatasetDict, IterableDataset]): The dataset used for initializing the index.
        index_k (int): The number of nearest neighbors to consider in the index.
        sentence_transformer_name (str): The name of the sentence transformer model.
        sentence_transformer_dim (str): The dimension of the sentence transformer embeddings.
        cache_dir (str): Directory to cache the sentence transformer model.
        sentence_transformer_batch_size (int): Batch size for processing with the sentence transformer.

    """
    
    dataset = dataset_module["train_dataset"]
    index_dimension = finetuning_args.sentence_transformer_dim
    index_k = finetuning_args.index_k
    sentence_transformer_name = finetuning_args.sentence_transformer_name
    sentence_transformer_dim = finetuning_args.sentence_transformer_dim
    cache_dir = finetuning_args.sentence_transformer_cache_dir
    sentence_transformer_batch_size = finetuning_args.sentence_transformer_batch_size
        
    from .mome_basetuner import MoMEAttentionAdaptor
    for name, module in model.named_modules():
        if isinstance(module, MoMEAttentionAdaptor):
            module.update_layer(index_dimension=index_dimension)
            module.initialize_index(
                dataset=dataset, 
                index_k=index_k, 
                sentence_transformer_name=sentence_transformer_name, 
                sentence_transformer_dim=sentence_transformer_dim, 
                cache_dir=cache_dir, 
                sentence_transformer_batch_size=sentence_transformer_batch_size)



from concurrent.futures import ThreadPoolExecutor
from .mome_index import LaminiIndex   # 假设你的索引类名是这个

def build_shared_index(
    dataset, index_k, st_name, st_dim, cache_dir, batch_size
) -> LaminiIndex:
    index = LaminiIndex()
    index.initialize(
        dataset=iter(dataset),           # 变成一次性 iterator
        index_k=index_k,
        sentence_transformer_name=st_name,
        sentence_transformer_dim=st_dim,
        cache_dir=cache_dir,
        sentence_transformer_batch_size=batch_size,
    )
    return index

def find_and_initialize_mome_adapters(        
        model: nn.Module, 
        dataset_module: DatasetModule, 
        finetuning_args: "FinetuningArguments",
    ) -> None:
    from .mome_basetuner import MoMEAttentionAdaptor
    # ---- 1. 预计算一次索引 ----
    dataset = dataset_module["train_dataset"]
    shared_index = build_shared_index(
        dataset               = dataset,
        index_k               = finetuning_args.index_k,
        st_name               = finetuning_args.sentence_transformer_name,
        st_dim                = finetuning_args.sentence_transformer_dim,
        cache_dir             = finetuning_args.sentence_transformer_cache_dir,
        batch_size            = finetuning_args.sentence_transformer_batch_size,
    )

    # ---- 2. 找到所有 adaptor ----
    adapters: list[tuple[str, MoMEAttentionAdaptor]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoMEAttentionAdaptor)
    ]
    print(f"Found {len(adapters)} MoMEAttentionAdaptor layers")

    # ---- 3. 并行更新 config + 绑定共享索引 ----
    def _init_adapter(item):
        name, adaptor = item
        adaptor.update_layer(index_dimension=finetuning_args.sentence_transformer_dim)
        adaptor.set_prebuilt_index(shared_index)  # 你需要在 adaptor 里加一个 setter
        return name

    with ThreadPoolExecutor() as pool:            # CPU‑bound，不建议用 GPU 线程
        for layer_name in pool.map(_init_adapter, adapters):
            print(f"  ↳ adaptor '{layer_name}' initialized")

    # 如果你担心 Python GIL 可换 `ProcessPoolExecutor`，
    # 但需确保 LaminiIndex 支持进程序列化（Torch tensor share_memory 或拷贝）

