import torch.nn as nn


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
