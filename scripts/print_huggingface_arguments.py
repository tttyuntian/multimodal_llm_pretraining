from pprint import pprint

import tyro
from experiments.config import TrainingConfig


@tyro.conf.configure(tyro.conf.OmitArgPrefixes)
def get_arguments(micro_batch_size: int, gradient_accumulation_steps: int, config: TrainingConfig) -> dict:
    return config.training_class(
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )._to_huggingface_args_dict()


if __name__ == "__main__":
    hf_args = tyro.cli(get_arguments)
    pprint(hf_args)
