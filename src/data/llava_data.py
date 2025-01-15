import json
import os
from PIL import Image
from tqdm import tqdm

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


def load_llava_data(path_to_data, split="pretrain"): # split = pretrain / instruction
    if split == "pretrain":
        with open(os.path.join(path_to_data,"blip_laion_cc_sbu_558k.json"), "r") as f:
            llava_data = json.load(f)

        # llava_data = llava_data[:5000]  # TODO: remove

        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            # load the actual images
            llava_data[i]["image_path"] = os.path.join(path_to_data, "images", llava_data[i]["image"])
            llava_data[i]["conversations"] = process_conversations(llava_data[i]["conversations"])

    elif split == "instruction":

        with open(os.path.join(path_to_data,"llava_v1_5_mix665k.json"), "r") as f:
            llava_data = json.load(f)
        
        # llava_data = llava_data[:5000]  # TODO: remove

        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            # load the actual images
            llava_data[i]["image_path"] = os.path.join(path_to_data, llava_data[i]["image"])
            llava_data[i]["conversations"] = process_conversations(llava_data[i]["conversations"])

    else:
        raise NotImplementedError("data split not implemented")

    return llava_data


class LlavaDataset(Dataset):
    def __init__(
        self,
        path_to_llava_data="/gpfs/data/superlab/datasets/LLaVA-Pretrain",  # TODO: remove default arguments
        split="pretrain", # pretrain or instruction
    ) -> None:
        super().__init__()
        self._all_data = load_llava_data(path_to_llava_data, split=split)
        
    def __len__(self):
        return len(self._all_data)

    def get_image(self, idx):
        return Image.open(self._all_data[idx]["image_path"])
        # return self._all_data[idx]["image"]

    def __getitem__(self, idx):
        return {
            "image": self.get_image(idx),
            "conversations": self._all_data[idx]["conversations"],
        }


class LlavaCollator:
    def __init__(self):
        self.processor = LlavaProcessor(
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336").image_processor
        )
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.processor.tokenizer.add_tokens("<image>") # add the new `<image>` token

    def __call__(self, features, return_tensors=None):
        """
        Custom collate function to preprocess images and tokenize text in features (i.e., batches).

        Args:
            batch (list): A list of samples, where each sample is a dictionary
                        with keys "image" and "conversations".
            processor (LlavaProcessor): LlavaProcessor to process images and text.

        Returns:
            dict: A batch dictionary containing processed "pixel_values", "input_ids", and "labels".
        """
        # Extract images and conversations from the batch
        images = [item["image"] for item in features]
        conversations = [item["conversations"] for item in features]

        # Process images
        processed_images = self.processor.image_processor(images, return_tensors="pt")

        # Tokenize text using the processor's chat template
        tokenized_texts = [
            self.processor.tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]
        tokenized_features = self.processor.tokenizer(
            tokenized_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Create labels (deep copy of input_ids)
        labels = tokenized_features.input_ids.clone()

        return {
            "pixel_values": processed_images["pixel_values"],
            "input_ids": tokenized_features.input_ids,
            "attention_mask": tokenized_features.attention_mask,
            "labels": labels
        }
