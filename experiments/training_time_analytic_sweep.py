import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from experiments import Sweep
from experiments.config import BaseConfig
from experiments.training_time_analytic import TrainingTimeAnalytic


@dataclass
class TrainingTimeAnalyticSweep(Sweep[TrainingTimeAnalytic]):
    search_space: Annotated[dict[str, list] | Path | str, tyro.conf.arg(constructor=Path)]

    def __post_init__(self) -> None:
        if isinstance(self.search_space, (Path, str)):
            with open(self.search_space) as f:
                self.search_space: dict[str, list] = json.load(f)

    @property
    def experiments(self) -> list[TrainingTimeAnalytic]:
        return [
            TrainingTimeAnalytic(config=BaseConfig(**config_kwargs))
            for config_kwargs in self._kwargs_product(**self.search_space)
        ]


if __name__ == "__main__":
    TrainingTimeAnalyticSweep.cli()
