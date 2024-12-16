from typing import Any, Literal

import torch
import torch.optim
from transformers import (
    AutoConfig,
    CLIPVisionConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    PreTrainedModel,
    SchedulerType,
)

from transformers import AutoTokenizer, AutoProcessor


from . import LlavaT, MultimodalModelClass


class LlavaModelClass(MultimodalModelClass[LlavaT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336")
        text_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        config = LlavaConfig(
            vision_config=vision_config,
            text_config=text_config,
        )
        model = LlavaForConditionalGeneration(config) # TODO: check the forward() call and see how to introduce image special token.
        
        # Freeze visual encoder's parameters
        for name, param in model.named_parameters():
            if name.startswith("vision_tower"):
                param.requires_grad = False


        # add a new <image> token and change the config.image_token_index to that new token
        processor = LlavaProcessor(
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336").image_processor
        )

        processor.tokenizer.add_tokens("<image>")

        model.resize_token_embeddings(len(processor.tokenizer)) # need this to avoid cuda error
        model.config.image_token_index = processor.tokenizer.encode("<image>", add_special_tokens=False)[0]

        return model

    @property
    def supports_activation_checkpointing(self) -> bool:
        """Some models don't implement activation (aka gradient) checkpointing. Override and return False if so.
        Refer to PreTrainedModel.supports_gradient_checkpointing.
        You can also implement it yourself (see convnext.py for an example).
        """
        return True

    @property
    def supports_compilation(self) -> bool:
        """Some models do not support torch.compile. Override and return False if so."""
        return True

    @property
    def batch_size(self) -> int:
        """Overall batch size. In our scripts, (num_nodes * gpus_per_node * micro_batch_size * grad_acc_steps)
        always equals batch_size."""
        return 256

    @property
    def training_steps(self) -> int:
        """Total number of training steps."""
        return 2180

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        """Whether to used mixed precision. None if only fp32 precision."""
        return None

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        """The PyTorch optimizer class (not instantiated object), e.g. `torch.optim.AdamW`."""
        return torch.optim.AdamW

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the optimizer class. Not including `params`."""
        return {
            "lr": 1e-3,
            "weight_decay": 0.0,
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        """Learning rate scheduler, referring to implementations in HuggingFace Transformers.
        transformers.SchedulerType (https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.SchedulerType)"""
        return SchedulerType.COSINE

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for scheduler. Not including `optimizer` or `num_training_steps`."""
        return {
            "num_warmup_steps": int(self.training_steps * 0.03),
        }

    @property
    def max_grad_norm(self) -> float:
        """Maximum gradient norm (for gradient clipping)."""
        return 0.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        """Any extra hyper-parameters for transformers.TrainingArguments."""
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        """Name of modules to wrap as FSDP units. Usually the significant model layers, e.g. `['GPTNeoXLayer']`."""
        return ["LlamaDecoderLayer"]

    @property
    def image_size(self) -> int:
        return 336

    @property
    def vocab_size(self) -> int:
        return 128256

    @property
    def sequence_length(self) -> int:
        return 131072
