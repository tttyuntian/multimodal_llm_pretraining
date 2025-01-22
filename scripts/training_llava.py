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


def get_model(model_type: ModelT) -> PreTrainedModel:
    return get_model_class(model_type).build_model(use_custom_kernels=True)


def get_dataset(model_type: ModelT, data_path: Path, data_split: str) -> Dataset:
    from src.data.llava_data import LlavaDataset
    return LlavaDataset(
        path_to_llava_data=data_path,
        split=data_split,
    )


def get_data_collator(model_type: ModelT, patch_size: int, vision_feature_select_strategy: str):
    if model_type in ["llava-pretrain", "llava-finetune"]:
        from src.data.llava_data import LlavaCollator
        return LlavaCollator(patch_size, vision_feature_select_strategy)
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
):
    if check_cuda_p2p_ib_support() is False:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"

    model = get_model(model_type)
    train_dataset = get_dataset(model_type, data_path, data_split)
    data_collator = get_data_collator(
        model_type, 
        patch_size=model.config.vision_config.image_size,
        vision_feature_select_strategy="default",
    )

    optimizer_cls_and_kwargs = get_optimizer_cls_and_kwargs(
        model_type, using_deepspeed=(training_arguments.get("deepspeed") is not None)
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            **training_arguments,
            remove_unused_columns=False
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
        ),
    )


if __name__ == "__main__":
    tyro.cli(run)