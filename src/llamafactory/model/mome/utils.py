import json
import os
import re
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file

from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.hparams.training_args import TrainingArguments
from llamafactory.model.mome.mome_basetuner import MoMEAttentionAdaptor
from .mome_index import LaminiIndex 
from llamafactory.data.data_utils import DatasetModule
from collections.abc import MutableSequence
from typing import Iterable, List, Optional, Tuple, Dict
from ...extras import logging
from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftConfig


MOME_ADAPTER_CONFIG_FILE = "mome_adapter_config.json"
ADAPTER_MODEL_FILE = "adapter_model.safetensors"

logger = logging.get_logger(__name__)


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


def save_mome_adapter_config(adapter_save_path: str, finetuning_args: "FinetuningArguments"):
    """
    Save MoME adapter configuration into a JSON file.
    """
    mome_config = {
        "mome_index_k": finetuning_args.index_k,
        "mome_sentence_transformer_name": finetuning_args.sentence_transformer_name,
        "mome_sentence_transformer_dim": finetuning_args.sentence_transformer_dim,
        "mome_num_experts": finetuning_args.num_experts,
    }

    os.makedirs(adapter_save_path, exist_ok=True)
    mome_config_path = os.path.join(adapter_save_path, MOME_ADAPTER_CONFIG_FILE)
    
    with open(mome_config_path, "w") as f:
        json.dump(mome_config, f, indent=2)

    logger.info_rank0(f"Saved MoME adapter config to {mome_config_path}")
    
    
def find_and_initialize_mome_adapters(        
        model: nn.Module, 
        dataset_module: DatasetModule, 
        finetuning_args: "FinetuningArguments",
        training_args: "TrainingArguments",
    ) -> None:
    """
    Find and initialize MoME adapters in the model.
    """
    
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
    logger.info_rank0(f"Found {len(adapters)} MoMEAttentionAdaptor layers")
    if len(adapters) == 0:
        logger.warning_rank0("No MoMEAttentionAdaptor layers found in the model")
        return


    # ---- 3. Parallel update config + bind shared index ----
    def _init_adapter(item):
        name, adaptor = item
        # adaptor.update_layer(index_dimension=finetuning_args.sentence_transformer_dim)
        adaptor.initialize_index_from_prebuilt_index(shared_index) 
        return name

    with ThreadPoolExecutor() as pool:
        for layer_name in pool.map(_init_adapter, adapters):
            logger.info_rank0(f"  ↳ adaptor '{layer_name}' initialized")
    
    # num_experts is coupled with the size of the dataset, so read the num here
    finetuning_args.num_experts = shared_index.values.shape[0]
    
    save_mome_adapter_config(adapter_save_path=training_args.output_dir, finetuning_args=finetuning_args)



ATTN_NAME_KEYWORDS = {
    # Hugging Face attention class names that appear in the most common OSS LLMs
    "self_attn",
    "attn",
    "attention",                   # generic fallback
    "llamaattention",              # Llama / Mistral-style
    "mistralattention",
    "qwenattention",               # Qwen-style
    "deepseekattention",           # DeepSeek-style
    "gemmaattention",              # Gemma-style
    "falconattention",             # Falcon-style
    "xverseattention",             # Baichuan / Xverse-style
}

def _is_attention_layer(m: nn.Module) -> bool:
    """
    Heuristic to decide whether `m` is a self-attention layer that should be wrapped.
    The rule is purposely forgiving so that it works for most open-source LLM repos
    without hard-coding every class.
    """
    cls_name = m.__class__.__name__.lower()
    if isinstance(m, MoMEAttentionAdaptor):                   # already wrapped
        return False
    if any(key in cls_name for key in ATTN_NAME_KEYWORDS):
        return True
    # Fallback: check for common proj attributes
    has_qk = all(hasattr(m, proj) for proj in ("q_proj", "k_proj", "v_proj"))
    return has_qk


def _replace_module(parent: nn.Module, child_name: str, child_mod: nn.Module,
                    adaptor_factory) -> None:
    """
    Replaces `parent.child_name` with the object returned by `adaptor_factory`.
    Handles both attribute and list-like containers (ModuleList / Sequential).
    """
    wrapped = adaptor_factory(child_mod)

    # Attribute assignment covers almost all cases …
    if hasattr(parent, child_name):
        setattr(parent, child_name, wrapped)
        return

    # … but ModuleList / Sequential store modules by **index**, not attribute.
    if isinstance(parent, MutableSequence):
        parent[int(child_name)] = wrapped
        return

    raise RuntimeError(f"Cannot replace child {child_name} of type {type(parent)}.")


def wrap_attn_with_mome(
    model: nn.Module,
    adapter_name: str = "default",
    r: int = 8,
    lora_alpha: int = 8,
    index_dimension: int = 768,
    wrap_num: Optional[int] = None,
) -> nn.Module:
    """
    Recursively traverses `model`, finds all attention layers that satisfy
    `_is_attention_layer`, and wraps each of them with `MoMEAttentionAdaptor`.
    
    Parameters
    ----------
    model : nn.Module
        The LLM returned by `from_pretrained(...)`.
    adapter_name : str, optional
        Name registered inside every `MoMEAttentionAdaptor`.  Defaults to "default".
    r : int, optional
        LoRA rank.  Defaults to 8.
    lora_alpha : int, optional
        LoRA alpha / scaling factor.  Defaults to 8.
    wrap_num : int, optional
        Number of attention layers from beginning to wrap. If None, wraps all attention layers.
    
    Returns
    -------
    list[str]
        A list of dotted module-paths that were replaced.
    """
    
    replaced_paths: List[str] = []

    def adaptor_factory(base_layer: nn.Module) -> MoMEAttentionAdaptor:
        return MoMEAttentionAdaptor(
            base_layer=base_layer,
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            index_dimension=index_dimension,
        )

    # (parent_mod, child_name, child_mod, dotted_path)
    stack: List[Tuple[nn.Module, str, nn.Module, str]] = []

    # Build traversal stack
    for mod_name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            dotted = f"{mod_name}.{child_name}" if mod_name else child_name
            stack.append((mod, child_name, child_mod, dotted))

    # Replace in reverse depth order so children are wrapped before parents iterate further
    for parent, child_name, child_mod, dotted in reversed(stack):
        if _is_attention_layer(child_mod):
            if wrap_num is None or len(replaced_paths) < wrap_num:
                _replace_module(parent, child_name, child_mod, adaptor_factory)
                replaced_paths.append(dotted)
            else:
                break

    logger.info_rank0(f"Wrapped {len(replaced_paths)} attention layers")
    logger.info_rank0(f"Replaced paths: {replaced_paths}")
    return model

def _normalise_key(key: str) -> str:
    """
    Strip the extra prefixes that PEFT adds so the key matches
    `model.state_dict()` names.
      base_model.model.model.layers.X…  ->  model.layers.X…
    """
    key = re.sub(r"^base_model\.", "", key)          # drop leading 'base_model.'
    key = re.sub(r"^model\.model\.", "model.", key)  # collapse duplicate 'model.'
    return key


def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], 
    adapter_name: str = "default", 
    parameter_prefix: str = "lora_"
) -> dict[str, torch.Tensor]:
    """
    Utility function to remap the state_dict keys to fit the PEFT model by inserting the adapter name.
    Example:
        'base_model.model.model.layers.0.mlp.down_proj.lora_A.weight'
        ->
        'base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight'
    
    """
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict

def _find_self_attn_module_path(param_key: str) -> str | None:
    """
    >>> _find_self_attn_module_path(
    ...     "model.layers.12.self_attn.lora_attn_query_in.default.weight")
    'model.layers.12.self_attn'
    """
    if ".self_attn." not in param_key:
        return None
    return param_key.split(".self_attn.", 1)[0] + ".self_attn"

def load_mome_weights(
    model: torch.nn.Module,
    safetensors_path: str,
    adapter_name: str = "default",
) -> Tuple[Iterable[str], Iterable[str], Iterable[str]]:
    """
    Load **only** the weights that belong to MoMEAttentionAdaptor modules.

    Parameters
    ----------
    model : torch.nn.Module
        The instantiated Llama-family model that already contains
        MoMEAttentionAdaptor wrappers.
    safetensors_path : str
        Path to the checkpoint file *or* to a directory containing one.

    Returns
    -------
    loaded_keys, missing_keys, unexpected_keys : tuple of iterables
        Exactly what `load_state_dict` returns, so you can log / assert.
    """
    safetensors_file = os.path.join(safetensors_path, ADAPTER_MODEL_FILE)
    raw_state: Dict[str, torch.Tensor] = load_file(safetensors_file)
    
    raw_state = _insert_adapter_name_into_state_dict(
        raw_state, adapter_name=adapter_name
    )

    model_state_keys = set(model.state_dict().keys())
    named_modules = dict(model.named_modules())

    filtered_state: Dict[str, torch.Tensor] = {}

    for raw_key, tensor in raw_state.items():
        key = _normalise_key(raw_key)

        # (1) must exist in the live model
        if key not in model_state_keys:
            continue

        # (2) only parameters under .self_attn.*
        self_attn_path = _find_self_attn_module_path(key)
        if self_attn_path is None:
            continue

        # (3) the `.self_attn` module itself must be a MoMEAttentionAdaptor
        mome_module = named_modules.get(self_attn_path)
        if mome_module is None or mome_module.__class__.__name__ != "MoMEAttentionAdaptor":
            continue

        filtered_state[key] = tensor


    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    return tuple(filtered_state.keys()), missing, unexpected


def load_adapter_config_and_wrap_attn(
    model: nn.Module,
    adapter_path: str | list[str], 
    is_trainable: bool, 
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    adapter_name: str = "default",
) -> nn.Module:
    init_kwargs = {
        "subfolder": model_args.adapter_folder,
        "offload_folder": model_args.offload_folder,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    
    # Handle case where adapter_path is a list
    if isinstance(adapter_path, list):
        adapter_path = adapter_path[-1]  # Use the last adapter in the list
    
    
    # load adapter config
    mome_config_path = os.path.join(adapter_path, MOME_ADAPTER_CONFIG_FILE)
    mome_config = json.load(open(mome_config_path))
    
    # LoraConfig
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig._get_peft_type(
                adapter_path,
                subfolder=init_kwargs.get("subfolder", None),
                revision=init_kwargs.get("revision", None),
                cache_dir=init_kwargs.get("cache_dir", None),
                use_auth_token=init_kwargs.get("use_auth_token", None),
                token=init_kwargs.get("token", None),
            )
        ].from_pretrained(adapter_path, **init_kwargs)
    config.inference_mode = not is_trainable
    
    config.mome_index_k = mome_config["mome_index_k"]
    config.mome_sentence_transformer_name = mome_config["mome_sentence_transformer_name"]
    config.mome_sentence_transformer_dim = mome_config["mome_sentence_transformer_dim"]
    config.mome_num_experts = mome_config["mome_num_experts"]
    
    
    model = wrap_attn_with_mome(
        model = model,
        adapter_name = adapter_name,
        r = config.r,
        lora_alpha = config.lora_alpha,
        index_dimension = config.mome_sentence_transformer_dim,
        wrap_num = None,
    )
    
    lamini_index_list: list[tuple[str, LaminiIndex]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LaminiIndex)
    ]
    
    for name, lamini_index in lamini_index_list:
        lamini_index.initialize_empty_weights(
            index_k=config.mome_index_k,
            sentence_transformer_dim=config.mome_sentence_transformer_dim,
            num_experts=config.mome_num_experts,
        )
    
    filtered_state, missing, unexpected = load_mome_weights(model, adapter_path, adapter_name)
    
    
    for i in filtered_state:
        if i not in missing:
            logger.info_rank0(f"Loaded {i}")
    
    return model