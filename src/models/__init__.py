"""Module for defining all supported models and their hyper-parameters.

Adding new models:
You should extend ModelT and get_model_class in this file.
Then implement BaseModelClass (via LanguageModelClass or VisionModelClass) in a new file in this directory.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar

import torch.optim
from torch.utils.data import Dataset
from transformers import PreTrainedModel, SchedulerType

from ..benchmarking.data import DummyImageClassificationDataset, DummyTextModelingDataset, DummyMultimodalLanguageModelingDataset

## Define and group model types

RobertaT = Literal["roberta"]

PythiaT = Literal[
    "pythia-14m",
    "pythia-31m",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]

MambaT = Literal["mamba"]

ConvNextT = Literal["convnext-large-1k", "convnext-large-22k", "convnext-xlarge-22k"]

ViTT = Literal["vit"]

LlavaT = Literal["llava-pretrain", "llava-finetune"]

ModelT = Literal[
    RobertaT,
    PythiaT,
    MambaT,
    ConvNextT,
    ViTT,
    LlavaT,
]

##

T = TypeVar("T", bound=ModelT)

class BaseModelClass(ABC, Generic[T]):
    """Define models and hyper-parameters using this class."""

    def __init__(self, model_type: T) -> None:
        self.model_type: T = model_type

    @abstractmethod
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        """Return transformers.PreTrainedModel corresponding to this class.

        Args:
            use_custom_kernels: Whether to use custom kernels for the model.
                E.g. For transformer models, PreTrainedConfig.from_pretrained(attn_implementation="sdpa") if supported
                Implementing the use_custom_kernels=False branch is optional (for benchmarking w/out free-lunch methods)
        """
        raise NotImplementedError

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
    @abstractmethod
    def batch_size(self) -> int:
        """Overall batch size. In our scripts, (num_nodes * gpus_per_node * micro_batch_size * grad_acc_steps)
        always equals batch_size."""
        raise NotImplementedError

    @property
    @abstractmethod
    def training_steps(self) -> int:
        """Total number of training steps."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        """Whether to used mixed precision. None if only fp32 precision."""
        raise NotImplementedError

    @property
    @abstractmethod
    def optimizer(self) -> type[torch.optim.Optimizer]:
        """The PyTorch optimizer class (not instantiated object), e.g. `torch.optim.AdamW`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def optimizer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the optimizer class. Not including `params`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def scheduler_type(self) -> SchedulerType:
        """Learning rate scheduler, referring to implementations in HuggingFace Transformers.
        transformers.SchedulerType (https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.SchedulerType)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def scheduler_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for scheduler. Not including `optimizer` or `num_training_steps`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_grad_norm(self) -> float:
        """Maximum gradient norm (for gradient clipping)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def hf_training_args(self) -> dict[str, Any]:
        """Any extra hyper-parameters for transformers.TrainingArguments."""
        raise NotImplementedError

    @property
    @abstractmethod
    def fsdp_layers_to_wrap(self) -> list[str]:
        """Name of modules to wrap as FSDP units. Usually the significant model layers, e.g. `['GPTNeoXLayer']`."""
        raise NotImplementedError

    @abstractmethod
    def load_dummy_dataset(self) -> Dataset:
        """torch.utils.Dataset corresponding to dummy data for this model."""
        raise NotImplementedError


class LanguageModelClass(Generic[T], BaseModelClass[T]):
    """Extension of BaseModelClass for language models.
    Provides dummy dataset implementation for language modeling objective."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        raise NotImplementedError

    def load_dummy_dataset(self) -> Dataset:
        """Specific objective for language modeling. Could override for other text objectives."""
        return DummyTextModelingDataset(vocab_size=self.vocab_size, sequence_length=self.sequence_length)


class VisionModelClass(Generic[T], BaseModelClass[T]):
    """Extension of BaseModelClass for vision models.
    Provides dummy dataset implementation for image classification objective."""

    @property
    @abstractmethod
    def image_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    def load_dummy_dataset(self) -> Dataset:
        """Specific objective for image classification. Could override for other vision objectives."""
        return DummyImageClassificationDataset(image_size=self.image_size, num_classes=self.num_classes)


class MultimodalModelClass(Generic[T], BaseModelClass[T]):
    """Extension of BaseModelClass for vision models.
    Provides dummy dataset implementation for image classification objective."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def image_size(self) -> int:
        raise NotImplementedError

    def load_dummy_dataset(self, sequence_length=512) -> Dataset:
        """Specific objective for image classification. Could override for other vision objectives."""
        return DummyMultimodalLanguageModelingDataset(
            vocab_size=self.vocab_size, 
            sequence_length=sequence_length, 
            image_size=self.image_size,
            image_token_id=self.image_token_index,
        )


def get_model_class(model_type: ModelT) -> BaseModelClass:
    match model_type:
        case "roberta":
            from .roberta import RobertaModelClass

            return RobertaModelClass(model_type)
        case (
            "pythia-14m"
            | "pythia-31m"
            | "pythia-70m"
            | "pythia-160m"
            | "pythia-410m"
            | "pythia-1b"
            | "pythia-1.4b"
            | "pythia-2.8b"
            | "pythia-6.9b"
            | "pythia-12b"
        ):
            from .pythia import PythiaModelClass

            return PythiaModelClass(model_type)
        case "mamba":
            from .mamba import MambaModelClass

            return MambaModelClass(model_type)
        case "convnext-large-1k" | "convnext-large-22k" | "convnext-xlarge-22k":
            from .convnext import ConvNextModelClass

            return ConvNextModelClass(model_type)
        case "vit":
            from .vit import ViTModelClass

            return ViTModelClass(model_type)
        case "llava-pretrain":
            from .llava import LlavaPretrainModelClass

            return LlavaPretrainModelClass(model_type)
        case "llava-finetune":
            from .llava import LlavaFinetuneModelClass

            return LlavaFinetuneModelClass(model_type)


__all__ = ["ModelT", "BaseModelClass", "LanguageModelClass", "VisionModelClass", "MultimodalModelClass", "get_model_class"]
