from typing import Any, Literal

import torch.optim
from transformers import (
    PreTrainedModel,
    RobertaConfig,
    RobertaForMaskedLM,
    SchedulerType,
)

from . import LanguageModelClass, RobertaT


class RobertaModelClass(LanguageModelClass[RobertaT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        config = RobertaConfig.from_pretrained("roberta-large", attn_implementation="eager")
        model = RobertaForMaskedLM(config)
        return model

    @property
    def batch_size(self) -> int:
        return 8192

    @property
    def training_steps(self) -> int:
        return 500000

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        return "fp16"

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        return torch.optim.Adam

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "lr": 4e-4,
            "betas": (0.9, 0.98),
            "weight_decay": 0.01,
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        return SchedulerType.LINEAR

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        return {"num_warmup_steps": 30_000}

    @property
    def max_grad_norm(self) -> float:
        return 0.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        return ["RobertaLayer"]

    @property
    def vocab_size(self) -> int:
        return 50265

    @property
    def sequence_length(self) -> int:
        return 512
