import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from experiments import Sweep
from experiments.config import TrainingConfig
from experiments.training_time_empirical import TrainingTimeEmpirical


@dataclass
class TrainingTimeEmpiricalSweep(Sweep[TrainingTimeEmpirical]):
    search_space: Annotated[dict[str, list] | Path | str, tyro.conf.arg(constructor=Path)]
    benchmarking_steps: int = 3
    trial: int = 0
    phase: int = None

    def __post_init__(self) -> None:
        if isinstance(self.search_space, (Path, str)):
            with open(self.search_space) as f:
                self.search_space: dict[str, list] = json.load(f)

    @property
    def experiments(self) -> list[TrainingTimeEmpirical]:
        experiment_list = []

        for config_kwargs in self._kwargs_product(**self.search_space):
            config = TrainingConfig(**config_kwargs)
            experiment = TrainingTimeEmpirical(
                config=config,
                benchmarking_steps=self.benchmarking_steps,
                trial=self.trial,
                phase=self.phase,
            )
            if experiment.is_valid():
                experiment_list.append(experiment)

        return experiment_list


if __name__ == "__main__":
    TrainingTimeEmpiricalSweep.cli()
