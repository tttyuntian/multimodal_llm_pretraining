import json
import os
from PIL import Image
from tqdm import tqdm
import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor
from torch.nn.utils.rnn import pad_sequence
import torchvision

from transformers import ViltProcessor

IGNORE_INDEX = -100


# CIFAR_CAPTION_TEMPLATES = [
#     'a photo of a {}.',
#     'a blurry photo of a {}.',
#     'a black and white photo of a {}.',
#     'a low contrast photo of a {}.',
#     'a high contrast photo of a {}.',
#     'a bad photo of a {}.',
#     'a good photo of a {}.',
#     'a photo of a small {}.',
#     'a photo of a big {}.',
#     'a photo of the {}.',
#     'a blurry photo of the {}.',
#     'a black and white photo of the {}.',
#     'a low contrast photo of the {}.',
#     'a high contrast photo of the {}.',
#     'a bad photo of the {}.',
#     'a good photo of the {}.',
#     'a photo of the small {}.',
#     'a photo of the big {}.',
# ]

CIFAR_CAPTION_TEMPLATES = [
    "An image of a {}"
]


def process_conversations_vilt_pretrain(conversations):
    
    return conversations[-1]["value"]

def process_conversations_vilt_finetune(conversations):
    results = []
    for line in conversations:
        results.append({
            "role": "assistant" if line["from"] == "gpt" else "user",
            "content": "".join(line["value"].split("<image>\n")),
        })
    
    return results


def load_llava_data(path_to_data, split):
    if split == "pretrain":
        with open(os.path.join(path_to_data,"blip_laion_cc_sbu_558k.json"), "r") as f:
            llava_data = json.load(f)

        for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
            llava_data[i]["image_path"] = os.path.join(path_to_data, "images", llava_data[i]["image"])
            llava_data[i]["caption"] = process_conversations_vilt_pretrain(llava_data[i]["conversations"])

    elif split == "instruction":

            with open(os.path.join(path_to_data,"llava_v1_5_mix665k.json"), "r") as f:
                llava_data = json.load(f)

            missing_image_idx_list = set([])
            for i in tqdm(range(len(llava_data)), total=len(llava_data), desc="Loading images and modifying conversations"):
                if "image" not in llava_data[i]:
                    missing_image_idx_list.add(i)
                    continue
                llava_data[i]["image_path"] = os.path.join(path_to_data, llava_data[i]["image"])
                llava_data[i]["conversations"] = process_conversations_vilt_finetune(llava_data[i]["conversations"])

            # Filter out the examples with no corresponding images
            llava_data = [
                example
                for i, example in enumerate(llava_data)
                if i not in missing_image_idx_list
            ]        

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
        self.split = split
        
    def __len__(self):
        return len(self._all_data)

    def get_image(self, idx):
        return Image.open(self._all_data[idx]["image_path"])

    def get_false_image(self, idx):
        current_idx = idx
        while current_idx == idx:
            current_idx = random.randint(0, self.__len__() - 1)
        return Image.open(self._all_data[current_idx]["image_path"])
    
    def __getitem__(self, idx):
        if self.split == "pretrain":
            return self._pretrain__getitem__(idx)
        else:
            return self._instruction__getitem__(idx)

    def _instruction__getitem__(self, idx):

        num_turns = len(self._all_data[idx]["conversations"]) // 2

        rand_turn = random.randint(0, num_turns-1)

        return {
            "image": self.get_image(idx),
            "conversations": self._all_data[idx]["conversations"][rand_turn*2:rand_turn*2+2],
        }

    def _pretrain__getitem__(self, idx):
        return {
            "image": self.get_image(idx),
            "caption": self._all_data[idx]["caption"],
            "false_image": self.get_false_image(idx)
        }


class Cifar100DatasetforVilt(Dataset):
    def __init__(self, path_to_data) -> None:
        super().__init__()
        self._all_data = torchvision.datasets.CIFAR100(
            root=path_to_data,
            train=True,
            download=True,
        )

        self.classes = [" ".join(c.split("_")) for c in self._all_data.classes]
        
    def __len__(self):
        return len(self._all_data)

    def get_image(self, idx):
        return self._all_data[idx][0]

    def get_false_image(self, idx):
        current_idx = idx
        while self._all_data[current_idx][1] == self._all_data[idx][1]:
            current_idx = random.randint(0, self.__len__() - 1)
        return self._all_data[current_idx][0]
    
    def __getitem__(self, idx):

        current_class_label = self.classes[self._all_data[idx][1]]

        caption_template = random.choice(CIFAR_CAPTION_TEMPLATES)

        return {
            "image": self.get_image(idx),
            "class_label": current_class_label,
            "caption": caption_template.format(current_class_label),
            "false_image": self.get_false_image(idx)
        }


class ViltCollator:
    def __init__(self, image_size, split, mlm_probability=0.15, sep_token="<|reserved_special_token_2|>"):
        self.image_size = image_size
        self.mlm_probability = mlm_probability

        self.split = split

        self.image_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").image_processor
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.tokenizer.pad_token_id = 128002 # TODO: get rid of hard-coded padding tokens
        self.tokenizer.mask_token = "<|reserved_special_token_1|>"
        self.tokenizer.mask_token_id = 128003 # TODO: get rid of hard-coded padding tokens
        self.sep_token = sep_token

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

    def _get_labels_no_tensor(self, input_ids, all_masks):
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

        labels = deepcopy(input_ids)

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if all_masks[i][j] == 0:
                    labels[i][j] = IGNORE_INDEX

        return labels

    def __call__(self, features):
        if self.split == "pretrain":
            return self._pretrain__call__(features)
        elif self.split == "instruction":
            return self._instruction_call(features)
        else:
            raise NotImplementedError(f"{self.split} is not supported yet.")


    def _instruction__call__(self, features, return_tensors="pt", return_data_types=["mlm"]):

        batch_size = len(features)
        items_to_return = {}

        images = [item["image"] for item in features]
        pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']

        questions = [item["conversations"][0]['content'] for item in features]
        answers   = [item["conversations"][1]['content'] + self.sep_token for item in features]

        tokenized_questions = self.tokenizer(questions)
        tokenized_answers = self.tokenizer(answers, add_special_tokens=False)

        input_ids = [q + a for q, a in zip(tokenized_questions['input_ids'], tokenized_answers['input_ids'])]

        input_ids = [torch.tensor(seq) for seq in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        items_to_return.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids).long(),
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
            "labels": input_ids
        })


        if 'mlm' in return_data_types:
            # TODO: dynamically change the masking ratio

            questions = [item["conversations"][0]['content'] for item in features]
            answers   = [item["conversations"][1]['content'] + self.sep_token for item in features]

            tokenized_questions = self.tokenizer(questions)
            tokenized_answers = self.tokenizer(answers, add_special_tokens=False)

            subword_marked_texts = self._process_subwords(tokenized_answers)
            all_masks = []
            for s_marked_caption in subword_marked_texts:
                current_mask = self._whole_word_mask(s_marked_caption)
                all_masks.append(current_mask)
            
            # Prepare MLM Labels, -100 for non-masked tokens and input_id for masked tokens
            mlm_labels = self._get_labels_no_tensor(tokenized_answers['input_ids'], all_masks)

            mlm_tokenized_answers_input_ids = tokenized_answers['input_ids']

            # Replace Mask Positions with the Mask Token
            for i, mask in enumerate(all_masks):
                for j in range(len(mask)):
                    if mask[j] == 1:
                        mlm_tokenized_answers_input_ids[i][j] = self.tokenizer.mask_token_id

            mlm_input_ids = [q + a for q, a in zip(tokenized_questions['input_ids'], mlm_tokenized_answers_input_ids)]

            mlm_labels = [[IGNORE_INDEX for _ in q] + a for q, a in zip(tokenized_questions['input_ids'], mlm_labels)]

            mlm_input_ids = [torch.tensor(seq) for seq in mlm_input_ids]
            mlm_input_ids = pad_sequence(mlm_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            mlm_attention_mask = (mlm_input_ids != self.tokenizer.pad_token_id).long()

            mlm_labels = [torch.tensor(seq) for seq in mlm_labels]
            mlm_labels = pad_sequence(mlm_labels, batch_first=True, padding_value=IGNORE_INDEX)

            items_to_return.update({
                "mlm_input_ids": mlm_input_ids,
                "mlm_attention_mask": mlm_attention_mask,
                "mlm_token_type_ids": torch.zeros_like(mlm_input_ids).long(),
                "mlm_pixel_values": pixel_values,
                "mlm_pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
                "mlm_labels": mlm_labels, # mlm labels are input_ids for masked tokens and -100 anywhere else
            })

        return items_to_return
        
    def _pretrain__call__(self, features, return_tensors="pt", return_data_types=["mlm", "itm"]):

        batch_size = len(features)
        items_to_return = {}

        # =================== regular input part ======================

        # get images and captions
        images = [item["image"] for item in features]
        captions = [item["caption"] + self.sep_token for item in features]

        # tokenize captions
        pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']

        inputs = self.tokenizer(
            captions,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
        )

        batch_size = len(captions)
        device = inputs["input_ids"].device

        items_to_return.update({
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "token_type_ids": torch.zeros_like(inputs['input_ids']).long(),
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
            "labels": inputs['input_ids'],
        })

        # =================== MLM part ======================

        if "mlm" in return_data_types:
            mlm_input_ids = inputs['input_ids'].clone()

            subword_marked_texts = self._process_subwords(inputs)
            all_masks = []
            for s_marked_caption in subword_marked_texts:
                current_mask = self._whole_word_mask(s_marked_caption)
                all_masks.append(current_mask)
            
            # Prepare MLM Labels, -100 for non-masked tokens and input_id for masked tokens
            mlm_labels = self._get_labels(mlm_input_ids, all_masks)

            # Replace Mask Positions with the Mask Token
            for i, mask in enumerate(all_masks):
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
                mlm_input_ids[i, mask_tensor] = self.tokenizer.mask_token_id
            
            items_to_return.update({
                "mlm_input_ids": mlm_input_ids,
                "mlm_attention_mask": inputs['attention_mask'],
                "mlm_token_type_ids": torch.zeros_like(inputs['input_ids']).long(),
                "mlm_pixel_values": pixel_values,
                "mlm_pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
                "mlm_labels": mlm_labels, # mlm labels are input_ids for masked tokens and -100 anywhere else
            })
        
        # =================== ITM part ======================

        if "itm" in return_data_types:

            false_images = [item["false_image"] for item in features]

            # tokenize captions
            false_pixel_values = self.image_processor(false_images, return_tensors="pt")['pixel_values']

            # # Matched pairs (image-caption correctly aligned) get ITM label 1.
            matched_itm_labels = torch.ones(batch_size, dtype=torch.long, device=device)

            # # Construct mismatched pairs using the randomly sampled indices.
            mismatched_input_ids = inputs["input_ids"].clone()
            mismatched_attention_mask = inputs["attention_mask"].clone()
            mismatched_itm_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            mismatched_pixel_values = false_pixel_values

            # # combine matched and mismatched inputs to get itm inputs
            combined_input_ids = torch.cat([inputs["input_ids"], mismatched_input_ids], dim=0)
            combined_attention_mask = torch.cat([inputs["attention_mask"], mismatched_attention_mask], dim=0)
            combined_itm_labels = torch.cat([matched_itm_labels, mismatched_itm_labels], dim=0)
            combined_pixel_values = torch.cat([pixel_values, mismatched_pixel_values], dim=0)

            items_to_return.update({
                "itm_input_ids": combined_input_ids,
                "itm_attention_mask": combined_attention_mask,
                "itm_token_type_ids": torch.zeros_like(combined_input_ids).long(),
                "itm_pixel_values": combined_pixel_values,
                "itm_pixel_mask": torch.ones([2*batch_size, self.image_size, self.image_size]).long(),
                "itm_labels": combined_itm_labels, # for ITM loss, 1 for matched, 0 for mismatched
            })
            
        return items_to_return




class ViltCollator_for_original:
    def __init__(self, image_size, split, mlm_probability=0.15):

        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

        self.image_size = image_size
        self.mlm_probability = mlm_probability

        self.split = split

        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer


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
            if token == "[CLS]" or token == "[SEP]":
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

    def _get_labels_no_tensor(self, input_ids, all_masks):
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

        labels = deepcopy(input_ids)

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if all_masks[i][j] == 0:
                    labels[i][j] = IGNORE_INDEX

        return labels

    def __call__(self, features):
        if self.split == "pretrain":
            return self._pretrain__call__(features)
        elif self.split == "instruction":
            return self._instruction_call(features)
        else:
            raise NotImplementedError(f"{self.split} is not supported yet.")


    def _instruction__call__(self, features, return_tensors="pt", return_data_types=["mlm"]):

        batch_size = len(features)
        items_to_return = {}

        images = [item["image"] for item in features]
        pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']

        questions = [item["conversations"][0]['content'] for item in features]
        answers   = [item["conversations"][1]['content'] for item in features]

        tokenized_questions = self.tokenizer(questions)
        tokenized_answers = self.tokenizer(answers, add_special_tokens=False)

        input_ids = [q + a for q, a in zip(tokenized_questions['input_ids'], tokenized_answers['input_ids'])]

        input_ids = [torch.tensor(seq) for seq in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        items_to_return.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids).long(),
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
            "labels": input_ids
        })


        if 'mlm' in return_data_types:
            # TODO: dynamically change the masking ratio

            questions = [item["conversations"][0]['content'] for item in features]
            answers   = [item["conversations"][1]['content'] for item in features]

            tokenized_questions = self.tokenizer(questions)
            tokenized_answers = self.tokenizer(answers, add_special_tokens=False)

            subword_marked_texts = self._process_subwords(tokenized_answers)
            all_masks = []
            for s_marked_caption in subword_marked_texts:
                current_mask = self._whole_word_mask(s_marked_caption)
                all_masks.append(current_mask)
            
            # Prepare MLM Labels, -100 for non-masked tokens and input_id for masked tokens
            mlm_labels = self._get_labels_no_tensor(tokenized_answers['input_ids'], all_masks)

            mlm_tokenized_answers_input_ids = tokenized_answers['input_ids']

            # Replace Mask Positions with the Mask Token
            for i, mask in enumerate(all_masks):
                for j in range(len(mask)):
                    if mask[j] == 1:
                        mlm_tokenized_answers_input_ids[i][j] = self.tokenizer.mask_token_id

            mlm_input_ids = [q + a for q, a in zip(tokenized_questions['input_ids'], mlm_tokenized_answers_input_ids)]

            mlm_labels = [[IGNORE_INDEX for _ in q] + a for q, a in zip(tokenized_questions['input_ids'], mlm_labels)]

            mlm_input_ids = [torch.tensor(seq) for seq in mlm_input_ids]
            mlm_input_ids = pad_sequence(mlm_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            mlm_attention_mask = (mlm_input_ids != self.tokenizer.pad_token_id).long()

            mlm_labels = [torch.tensor(seq) for seq in mlm_labels]
            mlm_labels = pad_sequence(mlm_labels, batch_first=True, padding_value=IGNORE_INDEX)

            items_to_return.update({
                "mlm_input_ids": mlm_input_ids,
                "mlm_attention_mask": mlm_attention_mask,
                "mlm_token_type_ids": torch.zeros_like(mlm_input_ids).long(),
                "mlm_pixel_values": pixel_values,
                "mlm_pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
                "mlm_labels": mlm_labels, # mlm labels are input_ids for masked tokens and -100 anywhere else
            })

        return items_to_return
        
    def _pretrain__call__(self, features, return_tensors="pt", return_data_types=["mlm", "itm"]):

        batch_size = len(features)
        items_to_return = {}

        # =================== regular input part ======================

        # get images and captions
        images = [item["image"] for item in features]
        captions = [item["caption"] for item in features]

        # tokenize captions
        pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']

        inputs = self.tokenizer(
            captions,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
        )

        batch_size = len(captions)
        device = inputs["input_ids"].device

        items_to_return.update({
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "token_type_ids": torch.zeros_like(inputs['input_ids']).long(),
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
            "labels": inputs['input_ids'],
        })

        # =================== MLM part ======================

        if "mlm" in return_data_types:
            mlm_input_ids = inputs['input_ids'].clone()

            subword_marked_texts = self._process_subwords(inputs)
            all_masks = []
            for s_marked_caption in subword_marked_texts:
                current_mask = self._whole_word_mask(s_marked_caption)
                all_masks.append(current_mask)
            
            # Prepare MLM Labels, -100 for non-masked tokens and input_id for masked tokens
            mlm_labels = self._get_labels(mlm_input_ids, all_masks)

            # Replace Mask Positions with the Mask Token
            for i, mask in enumerate(all_masks):
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
                mlm_input_ids[i, mask_tensor] = self.tokenizer.mask_token_id
            
            class_labels = [item["class_label"] for item in features]
            
            answer_tokens = [self.tokenizer.encode(class_label, add_special_tokens=False) for class_label in class_labels]
            
            for i, answer_toks in enumerate(answer_tokens):
                # Convert answer tokens to a tensor (if not already)
                answer_toks_tensor = torch.tensor(answer_toks)
                # Create a boolean mask where the input token is one of the answer tokens
                mask_positions = torch.isin(mlm_input_ids[i], answer_toks_tensor)
                # Replace those positions with the mask token ID
                mlm_input_ids[i][mask_positions] = self.tokenizer.mask_token_id
                mlm_labels[i][mask_positions]    = inputs['input_ids'][i][mask_positions] 

            items_to_return.update({
                "mlm_input_ids": mlm_input_ids,
                "mlm_attention_mask": inputs['attention_mask'],
                "mlm_token_type_ids": torch.zeros_like(inputs['input_ids']).long(),
                "mlm_pixel_values": pixel_values,
                "mlm_pixel_mask": torch.ones([batch_size, self.image_size, self.image_size]).long(),
                "mlm_labels": mlm_labels, # mlm labels are input_ids for masked tokens and -100 anywhere else
            })
        
        # =================== ITM part ======================

        if "itm" in return_data_types:

            false_images = [item["false_image"] for item in features]

            # tokenize captions
            false_pixel_values = self.image_processor(false_images, return_tensors="pt")['pixel_values']

            # # Matched pairs (image-caption correctly aligned) get ITM label 1.
            matched_itm_labels = torch.ones(batch_size, dtype=torch.long, device=device)

            # # Construct mismatched pairs using the randomly sampled indices.
            mismatched_input_ids = inputs["input_ids"].clone()
            mismatched_attention_mask = inputs["attention_mask"].clone()
            mismatched_itm_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            mismatched_pixel_values = false_pixel_values

            # # combine matched and mismatched inputs to get itm inputs
            combined_input_ids = torch.cat([inputs["input_ids"], mismatched_input_ids], dim=0)
            combined_attention_mask = torch.cat([inputs["attention_mask"], mismatched_attention_mask], dim=0)
            combined_itm_labels = torch.cat([matched_itm_labels, mismatched_itm_labels], dim=0)
            combined_pixel_values = torch.cat([pixel_values, mismatched_pixel_values], dim=0)

            items_to_return.update({
                "itm_input_ids": combined_input_ids,
                "itm_attention_mask": combined_attention_mask,
                "itm_token_type_ids": torch.zeros_like(combined_input_ids).long(),
                "itm_pixel_values": combined_pixel_values,
                "itm_pixel_mask": torch.ones([2*batch_size, self.image_size, self.image_size]).long(),
                "itm_labels": combined_itm_labels, # for ITM loss, 1 for matched, 0 for mismatched
            })
            
        return items_to_return