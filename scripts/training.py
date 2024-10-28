"""A template Training script"""

import os

import torchrunx as trx
from accelerate.utils import check_cuda_p2p_ib_support
from src.models import ModelT, get_model_class
from torch.utils.data import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments


def prepare_environment():
    if check_cuda_p2p_ib_support() is False:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"


def get_trainer_cls() -> type[Trainer]:
    # If using Deepspeed, use Deepspeed's Adam optimizer
    if self.zero_stage == "0" or self.optimizer not in [torch.optim.Adam, torch.optim.AdamW]:

        class CustomOptimizerTrainer(Trainer):
            @staticmethod
            def get_optimizer_cls_and_kwargs(
                args: TrainingArguments, model=None
            ) -> tuple[type[torch.optim.Optimizer], dict[str, Any]]:
                return self.optimizer, self.optimizer_kwargs

            # Can remove this after transformers==4.43.0
            def create_optimizer(self):
                trainer_get_optimizer_fn = Trainer.get_optimizer_cls_and_kwargs
                Trainer.get_optimizer_cls_and_kwargs = self.get_optimizer_cls_and_kwargs
                optimizer = super().create_optimizer()
                Trainer.get_optimizer_cls_and_kwargs = trainer_get_optimizer_fn
                return optimizer

        return CustomOptimizerTrainer

    return Trainer


def get_model(model_type: ModelT) -> PreTrainedModel:
    return get_model_class(model_type).build_model(use_custom_kernels=True)


def get_dataset(model_type: ModelT) -> Dataset:
    return get_model_class(model_type).load_dummy_dataset()


def get_training_arguments() -> TrainingArguments:
    return TrainingArguments(
        output_dir="./output",
        **{},  # TODO: dictionary from scripts/print_huggingface_arguments.py
    )


def train():
    prepare_environment()

    trainer_cls = get_trainer_cls()

    model = get_model()
    train_dataset = get_dataset()
    training_args = get_training_arguments()

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    trx.launch(
        func=train,
        hostnames=["localhost"],  # TODO: change to list of nodes
        workers_per_host=4,  # TODO: change to GPUs-per-node
        log_dir=os.environ["TORCHRUNX_LOG_DIR"],
    )
