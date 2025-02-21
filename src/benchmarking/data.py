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


def substitute_mask_entries(tensor, mask_token, k):
    """
    Replaces k% of the entries in a tensor with a mask token.

    Args:
        tensor (torch.Tensor): The input tensor.
        mask_token (int/float/any type compatible with tensor): The token value to substitute.
        k (float): Fraction of entries to mask (e.g., 0.1 for 10%).
    
    Returns:
        torch.Tensor: A new tensor with k% of its entries replaced by mask_token.
    """
    # Clone tensor to avoid in-place modifications on the original tensor.
    tensor = tensor.clone()
    
    # Generate a random tensor of the same shape with values in [0, 1).
    random_tensor = torch.rand_like(tensor, dtype=torch.float)
    
    # Create a mask where True indicates that the random value is less than k.
    mask = random_tensor < k
    
    # Substitute the selected entries with the mask token.
    tensor[mask] = mask_token
    return tensor

class DummyMultimodalLanguageModelingForViltDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        image_size: int,
        num_samples: int = 20_000,
        percentage_masked = 0.15,
        mask_token = 128255,
    ) -> None:
        super().__init__()
        print(f"vocab_size: {vocab_size}")
        print(f"sequence_length: {sequence_length}")
        print(f"image_size: {image_size}")
        print(f"num_samples: {num_samples}")
        
        self.image_size = image_size
        self.input_ids = torch.randint(0, vocab_size, (num_samples, sequence_length))
        self.images = torch.rand((num_samples, 3, image_size, image_size))
        
        ### mlm:

        random_tensor = torch.rand_like(self.input_ids, dtype=torch.float)
        mask = random_tensor < percentage_masked
        
        self.mlm_input_ids = self.input_ids.clone()
        self.mlm_input_ids[mask] = mask_token

        self.mlm_labels = self.input_ids.clone()
        self.mlm_labels[~mask] = -100


        ### itm:

        self.itm_labels = (torch.rand(num_samples) < 0.5).long()


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": torch.ones_like(self.input_ids[idx]).long(),
            "token_type_ids": torch.zeros_like(self.input_ids[idx]).long(),
            "pixel_values": self.images[idx],
            "pixel_mask": torch.ones([self.image_size, self.image_size]).long(),
            "labels": self.input_ids[idx],

            "mlm_input_ids": self.mlm_input_ids[idx],
            "mlm_attention_mask": torch.ones_like(self.input_ids[idx]).long(),
            "mlm_token_type_ids": torch.zeros_like(self.input_ids[idx]).long(),
            "mlm_pixel_values": self.images[idx],
            "mlm_pixel_mask": torch.ones([self.image_size, self.image_size]).long(),
            "mlm_labels": self.mlm_labels[idx],
            
            "itm_input_ids": self.input_ids[idx],
            "itm_attention_mask": torch.ones_like(self.input_ids[idx]).long(),
            "itm_token_type_ids": torch.zeros_like(self.input_ids[idx]).long(),
            "itm_pixel_values": self.images[idx],
            "itm_pixel_mask": torch.ones([self.image_size, self.image_size]).long(),
            "itm_labels": self.itm_labels[idx],
        }

