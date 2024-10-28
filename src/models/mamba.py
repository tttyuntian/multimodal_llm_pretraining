from typing import Any, Literal

import torch.optim
from transformers import (
    MambaConfig,
    MambaForCausalLM,
    PreTrainedModel,
    SchedulerType,
)
from transformers.models.mamba import modeling_mamba

from . import LanguageModelClass, MambaT


class MambaModelClass(LanguageModelClass[MambaT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        config = MambaConfig.from_pretrained("state-spaces/mamba-2.8b-hf")
        model = MambaForCausalLM(config)

        if use_custom_kernels:
            assert modeling_mamba.is_fast_path_available
        else:
            modeling_mamba.is_fast_path_available = False

        return model

    @property
    def supports_compilation(self) -> bool:
        # https://github.com/huggingface/transformers/pull/31247
        return False

    @property
    def batch_size(self) -> int:
        return 128

    @property
    def training_steps(self) -> int:
        return 572_204

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        return "bf16"

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        return torch.optim.AdamW

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "lr": (1.6e-4) * 5,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        return SchedulerType.COSINE_WITH_MIN_LR

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "num_warmup_steps": int(0.1 * self.training_steps),
            "min_lr": 1e-5,
        }

    @property
    def max_grad_norm(self) -> float:
        return 1.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        return ["MambaBlock"]

    @property
    def vocab_size(self) -> int:
        return 50265

    @property
    def sequence_length(self) -> int:
        return 4096
