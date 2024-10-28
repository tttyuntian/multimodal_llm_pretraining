from dataclasses import dataclass
from typing import Any, Sequence

from src.benchmarking.flops import count_flops_per_example
from src.models import BaseModelClass, LanguageModelClass, ModelT, get_model_class
from tango import Step

from experiments import Experiment, SlurmJob, step
from experiments.config import GpuT


@step(cacheable=True, version="001")
def total_training_flops(model_name: ModelT) -> float:
    model_class: BaseModelClass = get_model_class(model_type=model_name)
    flops_per_example = count_flops_per_example(model_class=model_class)
    total_flops = flops_per_example * model_class.batch_size * model_class.training_steps
    return total_flops


@dataclass
class CountFlopsExperiment(Experiment):
    model: ModelT

    slurm_gpu: GpuT | None = None

    def __post_init__(self):
        model_class = get_model_class(model_type=self.model)
        self.num_training_examples = model_class.batch_size * model_class.training_steps
        if isinstance(model_class, LanguageModelClass):
            self.num_training_examples *= model_class.sequence_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
        }

    @property
    def step_dict(self) -> dict[str, Step]:
        return {
            "training_flops": total_training_flops(model_name=self.model),
        }

    @property
    def slurm_job(self) -> SlurmJob | None:
        return SlurmJob(
            time_min=20,
            num_nodes=1,
            mem_per_node=64,
            cpus_per_node=8,
            gpus_per_node=1,
            gpu_type=self.slurm_gpu,
        )

    @property
    def dependencies(self) -> Sequence[Experiment]:
        return []

    def results(self) -> dict:
        return {
            "num_training_examples": self.num_training_examples,
            "training_flops": self.step_result("training_flops"),
        }

    def print_results(self) -> None:
        print(self.results())


if __name__ == "__main__":
    CountFlopsExperiment.cli()
