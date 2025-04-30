import os
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union

import torch
from peft import PeftConfig, PeftModel
from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
from peft.utils.integrations import init_empty_weights
from transformers import PreTrainedModel


# If our adapters are standard nn.Modules, maybe we don't need this class
class MoMEPeftModel(PeftModel):
    def __init__(        
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ):
        # super().__init__(model, peft_config, adapter_name, autocast_adapter_dtype, low_cpu_mem_usage)
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                # TODO: add MOME TUNER here instead of LoraModel Tuner if necessary
                self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1


    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass


    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> PeftModel:
        pass 

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        # TODO: Revisit the logic of this to see if we need to override it
        return model

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        # low priority
        pass 

    def print_trainable_parameters(self) -> None:
        # low priority
        pass

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # low priority
        pass 

    @contextmanager
    def disable_adapter(self):
        pass 


    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        pass 


    def delete_adapter(self, adapter_name: str) -> None:
        pass 

    def load_adapter(
        self,
        model_id: Union[str, os.PathLike],
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ):
        pass 

    def set_adapter(self, adapter_name: str) -> None:
        pass 
    
    

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds base model info
        4. Adds quantization information if it was used
        """
        # low priority
        pass


class MoMEPeftModelForCausalLM(MoMEPeftModel):
    pass 


