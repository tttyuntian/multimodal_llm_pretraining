import dataclasses
from dataclasses import dataclass
from typing import Literal, get_args

from src.gpus import GpuT, ampere_or_newer_gpu
from src.models import BaseModelClass, ModelT, get_model_class
from src.train import TrainingClass

from experiments import TangoStringHash


@dataclass
class BaseConfig(TangoStringHash):
    num_nodes: int
    gpus_per_node: int
    gpu_type: GpuT
    model: ModelT

    def ampere_or_newer_gpu(self) -> bool:
        return ampere_or_newer_gpu(self.gpu_type)

    def model_class(self) -> BaseModelClass:
        return get_model_class(model_type=self.model)


@dataclass
class TrainingConfig(BaseConfig):
    free_lunch: bool = False

    activation_checkpointing: bool = False
    sharding: Literal[
        "",
        *(f"fsdp_{s}" for s in get_args(TrainingClass.FsdpShardingT) if s != "no_shard"),  # pyright: ignore
        *(f"zero_{s}" for s in get_args(TrainingClass.ZeroStageT) if s != "0"),  # pyright: ignore
    ] = ""
    offloading: bool = False

    def training_class(self, **training_class_overrides) -> TrainingClass:
        model_class = self.model_class()

        ## Free lunch

        if self.free_lunch:
            tf32 = self.ampere_or_newer_gpu()
            compile = model_class.supports_compilation
        else:
            tf32 = False
            compile = False

        ## Activation checkpointing

        activation_checkpointing = self.activation_checkpointing

        ## Sharding and offloading

        fsdp_sharding = "no_shard"
        fsdp_layers_to_wrap = []
        fsdp_offload = False

        zero_stage = "0"
        zero_offload_optimizer = False
        zero_offload_params = False

        if self.sharding.startswith("fsdp_"):
            fsdp_sharding = self.sharding[len("fsdp_") :]
            fsdp_layers_to_wrap = model_class.fsdp_layers_to_wrap
            if self.offloading:
                fsdp_offload = True
        elif self.sharding.startswith("zero_"):
            zero_stage = self.sharding[len("zero_") :]
            if self.offloading:
                zero_offload_optimizer = True
                if zero_stage in ["3", "3++"]:
                    zero_offload_params = True

        ##

        training_class = TrainingClass(
            num_training_steps=model_class.training_steps,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=activation_checkpointing,
            bf16=(model_class.mixed_precision == "bf16"),
            fp16=(model_class.mixed_precision == "fp16"),
            tf32=tf32,
            compile=compile,
            optimizer=model_class.optimizer,
            optimizer_kwargs=model_class.optimizer_kwargs,
            scheduler_type=model_class.scheduler_type,
            scheduler_kwargs=model_class.scheduler_kwargs,
            fsdp_sharding=fsdp_sharding,  # pyright: ignore [reportArgumentType]
            fsdp_layers_to_wrap=fsdp_layers_to_wrap,
            fsdp_offload=fsdp_offload,
            zero_stage=zero_stage,  # pyright: ignore [reportArgumentType]
            zero_offload_optimizer=zero_offload_optimizer,
            zero_offload_params=zero_offload_params,
            max_grad_norm=model_class.max_grad_norm,
            hf_training_args_overrides=model_class.hf_training_args,
        )
        training_class = dataclasses.replace(training_class, **training_class_overrides)
        return training_class
