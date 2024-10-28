from dataclasses import dataclass
from typing import Any, Sequence

from src.models import get_model_class
from tango import Step

from experiments import Experiment, SlurmJob, step
from experiments.config import BaseConfig
from experiments.count_flops import CountFlopsExperiment


@step(cacheable=True, version="002")
def estimate_training_days_from_flops(config: BaseConfig, training_flops: float) -> float:
    model_class = get_model_class(model_type=config.model)
    half_precision = model_class.mixed_precision is not None

    # H100 Datasheet: https://resources.nvidia.com/en-us-tensor-core
    # Datasheets for Ampere GPUs
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

    # without sparsity

    if half_precision:  # Peak BF16 Tensor TFLOPS
        match config.gpu_type:
            case "h100":
                gpu_tflops_per_sec = 756
            case "a100":
                gpu_tflops_per_sec = 312
            case "a6000":
                gpu_tflops_per_sec = 154.8
            case "geforce3090":
                gpu_tflops_per_sec = 71
            case _:
                raise NotImplementedError
    else:  # Peak TF32 Tensor TFLOPS
        match config.gpu_type:
            case "h100":
                gpu_tflops_per_sec = 378
            case "a100":
                gpu_tflops_per_sec = 156
            case "a6000":
                gpu_tflops_per_sec = 77.4
            case "geforce3090":
                gpu_tflops_per_sec = 35.6
            case _:
                raise NotImplementedError

    flops_per_sec = (config.num_nodes * config.gpus_per_node * gpu_tflops_per_sec) * 1e12
    flops_per_day = flops_per_sec * 60 * 60 * 24
    training_days = training_flops / flops_per_day

    return training_days


@dataclass
class TrainingTimeAnalytic(Experiment):
    config: BaseConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.config.num_nodes,
            "gpus_per_node": self.config.gpus_per_node,
            "gpu_type": self.config.gpu_type,
            "model": self.config.model,
        }

    @property
    def dependencies(self) -> Sequence[Experiment]:
        return [CountFlopsExperiment(model=self.config.model)]

    @property
    def step_dict(self) -> dict[str, Step]:
        training_flops = self.dependencies[0].step_dict["training_flops"]
        return {
            "training_flops": training_flops,
            "training_days": estimate_training_days_from_flops(config=self.config, training_flops=training_flops),
        }

    @property
    def slurm_job(self) -> SlurmJob | None:
        return None

    def results(self) -> dict:
        return {
            "training_days": self.step_result("training_days"),
        }

    def print_results(self) -> None:
        print(self.results())


if __name__ == "__main__":
    TrainingTimeAnalytic.cli()
