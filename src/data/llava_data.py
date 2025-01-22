from collections import defaultdict
import json
import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor

IGNORE_INDEX = -100


def process_conversations(conversations):
    results = []
    for line in conversations:
        results.append({
            "role": "assistant" if line["from"] == "gpt" else "user",
            "content": line["value"],
        })
    
    return results


def load_llava_data(path_to_data, split): # split = pretrain / instruction
    if split == "pretrain":
        with open(os.path.join(path_to_data,"blip_laion_cc_sbu_558k.json"), "r") as f:
            llava_data = json.load(f)

        # llava_data = llava_data[:5000]  # TODO: remove

        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            llava_data[i]["image_path"] = os.path.join(path_to_data, "images", llava_data[i]["image"])
            llava_data[i]["conversations"] = process_conversations(llava_data[i]["conversations"])

    elif split == "instruction":

        with open(os.path.join(path_to_data,"llava_v1_5_mix665k.json"), "r") as f:
            llava_data = json.load(f)
        
        # llava_data = llava_data[:5000]  # TODO: remove

        missing_image_idx_list = set([])
        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            if "image" not in llava_data[i]:
                missing_image_idx_list.add(i)
                continue
            llava_data[i]["image_path"] = os.path.join(path_to_data, llava_data[i]["image"])
            llava_data[i]["conversations"] = process_conversations(llava_data[i]["conversations"])

        # Filter out the examples with no corresponding images
        llava_data = [
            example
            for i, example in enumerate(llava_data)
            if i not in missing_image_idx_list
        ]        

    else:
        raise NotImplementedError("data split not implemented")

    return llava_data


class LlavaDataset(Dataset):
    def __init__(
        self,
        path_to_llava_data,
        split, # "pretrain" or "instruction"
    ) -> None:
        super().__init__()
        self._all_data = load_llava_data(path_to_llava_data, split=split)
        
    def __len__(self):
        return len(self._all_data)

    def get_image(self, idx):
        return Image.open(self._all_data[idx]["image_path"])

    def __getitem__(self, idx):
        return {
            "image": self.get_image(idx),
            "conversations": self._all_data[idx]["conversations"],
        }


class LlavaCollator:
    def __init__(self, patch_size, vision_feature_select_strategy="default"):
        self.processor = LlavaProcessor(
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336").image_processor,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        self.processor.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.processor.tokenizer.pad_token_id = 128002 # TODO: get rid of hard-coded padding tokens
        self.processor.tokenizer.add_tokens("<image>") # add the new `<image>` token

        self.eot_token_id = self.processor.tokenizer.eos_token_id
        self.assistant_token_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
        self.end_header_token_id = self.processor.tokenizer.encode("<|end_header_id|>", add_special_tokens=False)[0]

    def _get_tokenized_lens(self, conversations):
        all_tokenized_lens = []

        for conversation in conversations:

            curr_tokenized_lens = []
            
            for turn in conversation:
                if turn["role"] == "assistant":
                    tokenized_turn = self.processor.tokenizer(turn["content"], add_special_tokens=False)
                    curr_tokenized_lens.append(len(tokenized_turn["input_ids"]))
            
            all_tokenized_lens.append(curr_tokenized_lens.copy())
        
        return all_tokenized_lens

    def _process_end_header_candidates(self, inputs):
        res = defaultdict(list)
        for ex_id, pos_id in zip(inputs[0], inputs[1]):
            res[ex_id.item()].append(pos_id.item())
        return res

    def _get_end_header_candidates(self, input_ids):
        """ Find all the `<|end_header_id|>` tokens """
        res = torch.where(input_ids==torch.tensor(self.end_header_token_id))
        return self._process_end_header_candidates(res)
    
    def _get_assistance_seq_spans(self, input_ids, end_header_candidates, all_tokenized_lens):
        """ Find all the start and end positions of assistant sequences """
        all_assistant_seq_spans = []
        for ex_id, pos_id_list in end_header_candidates.items():
            assistant_span_id = 0
            for pos_id in pos_id_list:
                if input_ids[ex_id, pos_id - 1] == self.assistant_token_id:
                    span_len = all_tokenized_lens[ex_id][assistant_span_id]
                    all_assistant_seq_spans.append([ex_id, pos_id + 2, pos_id + 2 + span_len])
                    assistant_span_id += 1
        return all_assistant_seq_spans

    def _get_labels(self, input_ids, all_tokenized_lens):
        end_header_candidates = self._get_end_header_candidates(input_ids)
        all_assistant_seq_spans = self._get_assistance_seq_spans(input_ids, end_header_candidates, all_tokenized_lens)

        # Set everywhere to IGNORE_INDEX, except the assistant sequences
        labels = input_ids.clone()
        to_be_filtered = torch.ones_like(labels)
        for ex_id, start_id, end_id in all_assistant_seq_spans:
            to_be_filtered[ex_id, start_id:end_id] = 0

        to_be_filtered -= (input_ids == self.eot_token_id).int()
        labels = torch.where(to_be_filtered == 1, IGNORE_INDEX, labels)
        return labels

    def __call__(self, features, return_tensors=None):
        # Extract images and conversations from the batch
        images = [item["image"] for item in features]
        conversations = [item["conversations"] for item in features]

        # Get tokenized lengths of users' sequences
        all_tokenized_lens = self._get_tokenized_lens(conversations)

        # Tokenize text using the processor's chat template
        chat_texts = [
            self.processor.tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]

        inputs = self.processor(
            images=images, 
            text=chat_texts, 
            return_tensors="pt", 
            add_special_tokens=False,
            padding=True,
        )

        # Create labels (deep copy of input_ids)
        labels = self._get_labels(inputs["input_ids"], all_tokenized_lens)

        return {
            "pixel_values": inputs.pixel_values,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
        }
