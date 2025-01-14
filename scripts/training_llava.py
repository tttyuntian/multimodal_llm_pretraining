import json
import os
from pathlib import Path
from typing import Any

import torch.optim
import torchrunx
import tyro
from accelerate.utils import check_cuda_p2p_ib_support
from src.models import ModelT, get_model_class
from torch.utils.data import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments


def get_model(model_type: ModelT, phase: int) -> PreTrainedModel:
    return get_model_class(model_type).build_model(phase, use_custom_kernels=True)


def get_dataset(model_type: ModelT, data_path: Path, data_split: str) -> Dataset:
    from src.data.llava_data import LlavaDataset
    return LlavaDataset(
        path_to_llava_data=data_path,
        split=data_split,
    )


def get_data_collator(model_type: ModelT):
    if model_type == "llava":
        from src.data.llava_data import LlavaCollator
        return LlavaCollator()
    else:
        raise NotImplementedError(f"{model_type} has no data collator implemented yet.")


def get_optimizer_cls_and_kwargs(
    model_type: ModelT, using_deepspeed: bool
) -> tuple[type[torch.optim.Optimizer], dict[str, Any]] | None:
    model_class = get_model_class(model_type)

    if using_deepspeed and model_class.optimizer in [
        torch.optim.Adam,
        torch.optim.AdamW,
    ]:  # use Deepspeed's Adam optimizer
        return None

    return (model_class.optimizer, model_class.optimizer_kwargs)


def train(
    output_dir: str, 
    model_type: ModelT, 
    training_arguments: dict[str, Any],
    data_path: Path,
    data_split: str,
    phase: int,
):
    if check_cuda_p2p_ib_support() is False:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"

    model = get_model(model_type, phase)
    train_dataset = get_dataset(model_type, data_path, data_split)
    print(train_dataset.__getitem__(0), flush=True)
    data_collator = get_data_collator(model_type)

    optimizer_cls_and_kwargs = get_optimizer_cls_and_kwargs(
        model_type, using_deepspeed=(training_arguments.get("deepspeed") is not None)
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            **training_arguments,
            dataloader_num_workers=1,
        ),
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
    )

    trainer.train()


def run(
    launcher: torchrunx.Launcher, 
    output_dir: str, 
    model_type: ModelT, 
    training_arguments: Path, 
    data_path: Path, 
    data_split: str,
    phase: int,
):
    training_arguments = json.load(open(training_arguments, "r"))
    launcher.run(
        func=train,
        func_kwargs=dict(
            output_dir=output_dir,
            model_type=model_type,
            training_arguments=training_arguments,
            data_path=data_path,
            data_split=data_split,
            phase=phase,
        ),
    )


if __name__ == "__main__":
    tyro.cli(run)