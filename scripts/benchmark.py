import math
import signal
import sys
from typing import Literal

import tyro
from experiments import Sweep
from experiments.training_time_empirical_sweep import TrainingTimeEmpiricalSweep
from src.gpus import GpuT, ampere_or_newer_gpu
from src.models import ModelT, get_model_class


def validate_arguments(
    num_nodes: int,
    gpus_per_node: int,
    gpu_type: GpuT,
    model: ModelT,
):
    model_class = get_model_class(model)

    num_gpus = num_nodes * gpus_per_node

    assert (
        model_class.batch_size % num_gpus == 0
    ), f"model batch size ({model_class.batch_size}) should be evenly divisible by total GPUs ({num_gpus})"
    assert math.log2(
        model_class.batch_size // num_gpus
    ).is_integer(), f"batch size per gpu ({model_class.batch_size // num_gpus}) should be power of 2"

    if model_class.mixed_precision == "bf16":
        assert ampere_or_newer_gpu(gpu_type), "GPU must be ampere or newer to use mixed precision with bf16"


def run_benchmark(
    num_nodes: int,
    gpus_per_node: int,
    gpu_type: GpuT,
    model: ModelT,
    methods: Literal["naive", "free-lunch", "all"] = "all",
    cmd: Literal["run", "count", "print-incomplete", "print-results"] = "run",
    slurm: bool = False,
) -> None:
    validate_arguments(num_nodes, gpus_per_node, gpu_type, model)

    # Naive settings
    free_lunch = [False]
    activation_checkpointing = [False]
    sharding = [""]
    offloading = [False]

    if methods == "free-lunch":
        free_lunch = [True]
    elif methods == "all":
        free_lunch = [True]
        activation_checkpointing = [False, True]
        sharding = [
            "",
            "zero_1",
            "zero_2",
            "zero_3",
            "fsdp_shard_grad_op",
            "fsdp_full_shard",
        ]
        offloading = [False, True]

    experiment_sweep = TrainingTimeEmpiricalSweep(
        search_space=dict(
            num_nodes=[num_nodes],
            gpus_per_node=[gpus_per_node],
            gpu_type=[gpu_type],
            model=[model],
            free_lunch=free_lunch,
            activation_checkpointing=activation_checkpointing,
            sharding=sharding,
            offloading=offloading,
        )
    )

    Sweep.run(experiment_sweep=experiment_sweep, cmd=cmd, slurm=slurm)


if __name__ == "__main__":
    try:
        tyro.cli(run_benchmark)
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)
