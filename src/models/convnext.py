from typing import Any, Literal

import torch
import torch.optim
from torch.utils.checkpoint import checkpoint
from transformers import (
    ConvNextConfig,
    ConvNextForImageClassification,
    PreTrainedModel,
    SchedulerType,
)
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.models.convnext.modeling_convnext import ConvNextEncoder

from . import ConvNextT, VisionModelClass


class ConvNextModelClass(VisionModelClass[ConvNextT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        config_name = None
        match self.model_type:
            case "convnext-large-1k":
                config_name = "convnext-large-224"
            case "convnext-large-22k":
                config_name = "convnext-large-224-22k"
            case "convnext-xlarge-22k":
                config_name = "convnext-xlarge-224-22k"

        config = ConvNextConfig.from_pretrained(f"facebook/{config_name}")
        model = ConvNextForImageClassification(config)

        ## Manually adding support for activation checkpointing (in the style of PreTrainedModel).
        model.supports_gradient_checkpointing = True
        model.convnext.encoder.__class__ = ConvNextEncoderWithCheckpointing
        model.convnext.encoder.gradient_checkpointing = False  # pyright: ignore [reportArgumentType]
        ##

        return model

    @property
    def supports_activation_checkpointing(self) -> bool:
        # We add support for activation checkpointing manually!
        return True

    @property
    def batch_size(self) -> int:
        return 4096

    @property
    def training_steps(self) -> int:
        match self.model_type:
            case "convnext-large-1k":
                return 93600
            case "convnext-large-22k" | "convnext-xlarge-22k":
                return 311940

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        return None

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        return torch.optim.AdamW

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "lr": 4e-3,
            "betas": (0.9, 0.999),
            "weight_decay": 0.05,
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        return SchedulerType.COSINE

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        match self.model_type:
            case "convnext-large-1k":
                return {
                    "num_warmup_steps": 312 * 20,
                }
            case "convnext-large-22k":
                return {
                    "num_warmup_steps": 3466 * 5,
                }
            case "convnext-xlarge-22k":
                return {
                    "num_warmup_steps": 3466 * 5,
                }

    @property
    def max_grad_norm(self) -> float:
        return 0.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        return ["ConvNextStage"]

    @property
    def image_size(self) -> int:
        return 224

    @property
    def num_classes(self) -> int:
        match self.model_type:
            case "convnext-large-1k":
                return 1000
            case "convnext-large-22k" | "convnext-xlarge-22k":
                return 21841


class ConvNextEncoderWithCheckpointing(ConvNextEncoder):
    """ConvNextEncoder with modified forward for activation/gradient checkpointing."""

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
    ) -> tuple | BaseModelOutputWithNoAttention:
        all_hidden_states = ()

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            ## Modified
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer_module.__call__, hidden_states, use_reentrant=True)  # pyright: ignore [reportAssignmentType]
            else:
                hidden_states = layer_module(hidden_states)
            ##

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states or None] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,  # pyright: ignore [reportArgumentType]
        )
