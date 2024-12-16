import copy
import json
from PIL import Image
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor

def process_conversations(conversations):
    results = []
    for line in conversations:
        results.append({
            "role": "assistant" if line["from"] == "gpt" else "user",
            "content": line["value"]
        })
    
    return results

def load_LLava_data(path_to_data, split="pretrain"): # split = pretrain / finetune
    if split == "pretrain":
        with open(os.path.join(path_to_data,"blip_laion_cc_sbu_558k.json"), "r") as f:
            llava_data = json.load(f)

        # llava_data = llava_data[:5000]

        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            # load the actual images
            llava_data[i]["image"] = Image.open(os.path.join(path_to_data, "images" ,llava_data[i]["image"]))
            llava_data[i]["conversations"] = process_conversations(llava_data[i]["conversations"])

    else:
        raise NotImplementedError("instruction tuning data not implemented")

    return llava_data


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


class LlavaPretrainingDataset(Dataset):
    def __init__(
        self,
        path_to_llava_pretrain_data = "/gpfs/data/superlab/datasets/LLaVA-Pretrain"
    ) -> None:
        super().__init__()

        # Instantiate the LlavaProcessor specific to our models
        self.processor = LlavaProcessor(
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336").image_processor
        )
        self.processor.tokenizer.add_tokens("<image>") # add the new `<image>` token

        self._all_data = load_LLava_data(path_to_llava_pretrain_data)
        
    def __len__(self):
        return len(self._all_data)

    def __getitem__(self, index):

        sample = self.processor(
            images=self._all_data[index]["image"], 
            text=self.processor.tokenizer.apply_chat_template(
                self._all_data[index]["conversations"], 
                tokenize=False)
            )

        sample['labels'] = copy.deepcopy(sample["input_ids"])

        return sample