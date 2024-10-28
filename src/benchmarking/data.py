import copy

import torch
from torch.utils.data import Dataset


class DummyTextModelingDataset(Dataset):
    def __init__(self, vocab_size: int, sequence_length: int, num_samples: int = 50_000) -> None:
        super().__init__()
        self.input_ids = torch.randint(0, vocab_size, (num_samples, sequence_length))
        self.labels = copy.deepcopy(self.input_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }


class DummyImageClassificationDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        num_samples: int = 20_000,
    ) -> None:
        super().__init__()
        self.images = torch.rand((num_samples, 3, image_size, image_size))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "pixel_values": self.images[index],
            "labels": self.labels[index],
        }
