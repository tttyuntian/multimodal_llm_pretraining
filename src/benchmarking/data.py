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


class DummyMultimodalLanguageModelingDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        image_size: int,
        num_samples: int = 20_000,
        image_token_id: int = 32000,
    ) -> None:
        super().__init__()
        print(f"vocab_size: {vocab_size}")
        print(f"sequence_length: {sequence_length}")
        print(f"image_size: {image_size}")
        print(f"num_samples: {num_samples}")
        print(f"image_token_id: {image_token_id}")
        
        self.images = torch.rand((num_samples, 3, image_size, image_size))
        
        text_input_ids = torch.randint(0, vocab_size, (num_samples, sequence_length - 1))  # off-set by 1 for `<image>` token. WARNING: This token is llava-specific.
        self.input_ids = torch.cat((torch.tensor([image_token_id] * num_samples).unsqueeze(1), text_input_ids), dim=-1)
        self.labels = copy.deepcopy(self.input_ids)
        self.attention_mask = torch.ones(self.input_ids.shape, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "attention_mask": self.attention_mask[index],
            "pixel_values": self.images[index],
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }


class DummyMultimodalLanguageModelingForViltDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        image_size: int,
        num_samples: int = 20_000,
    ) -> None:
        super().__init__()
        print(f"vocab_size: {vocab_size}")
        print(f"sequence_length: {sequence_length}")
        print(f"image_size: {image_size}")
        print(f"num_samples: {num_samples}")
        
        self.images = torch.rand((num_samples, 3, image_size, image_size))
        
        self.input_ids = torch.randint(0, vocab_size, (num_samples, sequence_length - 1))  # off-set by 1 for `<image>` token. WARNING: This token is llava-specific.
        self.mlm_labels = copy.deepcopy(self.input_ids)
        self.itm_labels = torch.ones(num_samples, dtype=torch.long)
        self.attention_mask = torch.ones(self.input_ids.shape, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "pixel_values": self.images[index],
            "mlm_labels": self.mlm_labels[index],
            "itm_labels": self.itm_labels[index],
        }

