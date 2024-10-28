from typing import Any, Literal

import torch.optim
from transformers import PreTrainedModel, SchedulerType, ViTConfig, ViTForImageClassification

from . import VisionModelClass, ViTT


class ViTModelClass(VisionModelClass[ViTT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        config: ViTConfig = ViTConfig.from_pretrained(  # pyright: ignore [reportAssignmentType]
            "google/vit-large-patch16-224-in21k",
            num_labels=21841,
            hidden_dropout_prob=0.1,
            attn_implementation=("sdpa" if use_custom_kernels else "eager"),
        )
        return ViTForImageClassification(config=config)

    @property
    def batch_size(self) -> int:
        return 4096

    @property
    def training_steps(self) -> int:
        return 311948

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        return None

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        return torch.optim.Adam

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "weight_decay": 0.03,
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        return SchedulerType.LINEAR

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        return {"num_warmup_steps": 10000}

    @property
    def max_grad_norm(self) -> float:
        return 1.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        return ["ViTLayer"]

    @property
    def image_size(self) -> int:
        return 224

    @property
    def num_classes(self) -> int:
        return 21841
