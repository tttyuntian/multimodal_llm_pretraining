from collections import defaultdict
import json
import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor

IGNORE_INDEX = -100

def process_conversations_vilt(conversations):
    
    return conversations[-1]["value"]

def load_llava_data(path_to_data, split):
    if split == "pretrain":
        with open(os.path.join(path_to_data,"blip_laion_cc_sbu_558k.json"), "r") as f:
            llava_data = json.load(f)

        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            llava_data[i]["image_path"] = os.path.join(path_to_data, "images", llava_data[i]["image"])
            llava_data[i]["caption"] = process_conversations_vilt(llava_data[i]["conversations"])

    else:
        raise NotImplementedError("data split not implemented")

    return llava_data


class LlavaDatasetforVilt(Dataset):
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
            "caption": self._all_data[idx]["caption"],
        }




class ViltCollator:
    def __init__(self, image_processor, tokenizer, mlm_probability=0.15):

        self.mlm_probability = mlm_probability
        self.image_processor = image_processor
        self.tokenizer       = tokenizer
    

    def _process_subwords(self, tokenized_results):

        all_marked = []
        for i, sample in enumerate(range(len(tokenized_results['input_ids']))):
            tokens = tokenized_results.tokens(i)
            word_ids = tokenized_results.word_ids(i)

            marked_tokens = []

            current_word = -1
            for token, id in zip(tokens, word_ids):
                if id == None:
                    marked_tokens.append(token)
                else:
                    if id > current_word:
                        marked_tokens.append(token)
                        current_word += 1
                    else:  
                        marked_tokens.append(f"##{token}")
                        
            all_marked.append(marked_tokens)
        
        return all_marked


    # ref: https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/data/data_collator.py#L1028
    def _whole_word_mask(self, input_tokens, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            # if token == "[CLS]" or token == "[SEP]": # TODO: check these hard-coded tokens
            if token in ["<|begin_of_text|>", "<|eot_id|>"] or token.startswith("<|reserved_special_token"):
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels



    def _get_labels(self, input_ids, all_masks):
        """
        Create labels for masked language modeling.
        
        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token ids.
            all_masks (List[Union[torch.Tensor, List[bool]]]): List of boolean masks indicating which
                tokens have been masked. Each mask should be of shape (seq_len,), with True at positions
                where the token has been masked.
        
        Returns:
            torch.Tensor: A tensor of the same shape as input_ids where unmasked positions are replaced with -100,
                        so that the loss is computed only for the masked tokens.
        """

        labels = input_ids.clone()
        
        mask_tensor = torch.stack([
            torch.tensor(mask, dtype=torch.bool, device=input_ids.device)
            for mask in all_masks
        ])
        
        labels[~mask_tensor] = IGNORE_INDEX
        return labels


    def __call__(self, features, return_tensors="pt"):

        # --- 1. Extract Data ---
        images = [item["image"] for item in features]
        captions = [item["caption"] for item in features]

        # --- 2. Process images and Tokenize Captions  ---
        pixel_values = self.image_processor(images, return_tensors="pt")

        inputs = self.tokenizer(
            captions,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
        )
        
        # --- 3. Generate Whole Word Mask for MLM ---
        subword_marked_texts = self._process_subwords(inputs)
        all_masks = []
        for s_marked_caption in subword_marked_texts:
            current_mask = self._whole_word_mask(s_marked_caption)
            all_masks.append(current_mask)
        
        # --- 4. Prepare MLM Labels ---
        mlm_labels = self._get_labels(inputs["input_ids"], all_masks)

        # --- 5. Replace Mask Positions with the Mask Token ---
        for i, mask in enumerate(all_masks):
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=inputs["input_ids"].device)
            inputs["input_ids"][i, mask_tensor] = self.tokenizer.mask_token_id

        # --- 6. ITM Input Preparation: Create Matched and Random Mismatched Pairs ---
        batch_size = inputs["input_ids"].size(0)
        device = inputs["input_ids"].device

        # Matched pairs (image-caption correctly aligned) get ITM label 1.
        matched_itm_labels = torch.ones(batch_size, dtype=torch.long, device=device)

        # Create mismatched indices
        while True:
            mismatched_indices = torch.randperm(batch_size, device=device)
            if not torch.any(mismatched_indices == torch.arange(batch_size, device=device)):
                break

        # Construct mismatched pairs using the randomly sampled indices.
        mismatched_input_ids = inputs["input_ids"][mismatched_indices]
        mismatched_attention_mask = inputs["attention_mask"][mismatched_indices]
        mismatched_mlm_labels = mlm_labels[mismatched_indices]
        mismatched_itm_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        if pixel_values is not None:
            mismatched_pixel_values = pixel_values[mismatched_indices]
        else:
            mismatched_pixel_values = None

        # --- 7. Combine Matched and Mismatched Pairs ---
        combined_input_ids = torch.cat([inputs["input_ids"], mismatched_input_ids], dim=0)
        combined_attention_mask = torch.cat([inputs["attention_mask"], mismatched_attention_mask], dim=0)
        combined_mlm_labels = torch.cat([mlm_labels, mismatched_mlm_labels], dim=0)
        combined_itm_labels = torch.cat([matched_itm_labels, mismatched_itm_labels], dim=0)
        
        if pixel_values is None:
            combined_pixel_values = None
        else:
            combined_pixel_values = torch.cat([pixel_values, mismatched_pixel_values], dim=0)
        
        # --- 8. Return the Combined Tensors ---
        return {
            "pixel_values": combined_pixel_values,      
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "mlm_labels": combined_mlm_labels,          # for MLM loss
            "itm_labels": combined_itm_labels,          # for ITM loss, 1 for matched, 0 for mismatched
        }
