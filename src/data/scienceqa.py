from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import torch

class ScienceQADataset(Dataset):
    def __init__(self, split="validation"):

        self.dataset = load_dataset("derek-thomas/ScienceQA", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ScienceQAMultimodalDataCollator:
    def __init__(self, processor):
        """
        Args:
            processor: Hugging Face processor for multimodal models.
            max_length: Maximum token length for truncation and padding.
        """
        self.processor = processor


    def __call__(self, features):

        letters = "ABCDE"

        # only QAs that has an image will need an image token
        image_token_strs = ["<image>" if item["image"] != None else "" for item in features]

        choices_strs = []
        
        for f in features:
            # Format the choices to strings
            result = ", ".join([f"({letters[i]}) {choice}" for i, choice in enumerate(f["choices"])])
            choices_strs.append(result)

        questions = [f"{image_token_str}\nQuestion: {item['question']} Choices: {choice_str}. Answer with the option's letter from the given choices directly." for item, choice_str, image_token_str in zip(features, choices_strs, image_token_strs)]
        
        # image processor needs to take in a list of PIL images, thus substituting "None" to a blank image
        images = [item["image"] if item["image"] != None else Image.new("RGB", (336, 336), (0,0,0)) for item in features]
        answers = [item["answer"] for item in features]

        # Process images
        processed_images = self.processor.image_processor(images, return_tensors="pt")

        # Tokenize text using the processor's chat template
        tokenized_texts = [
            self.processor.tokenizer.apply_chat_template(
                [{
                    "role": "user", "content": q
                }], tokenize=False)
            for q in questions
        ]

        tokenized_features = self.processor.tokenizer(
            tokenized_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "pixel_values": processed_images["pixel_values"],
            "input_ids": tokenized_features.input_ids,
            "attention_mask": tokenized_features.attention_mask,
            "answers": torch.tensor(answers)
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor

    processor = LlavaProcessor(
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
        image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336").image_processor
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.add_tokens("<image>")

    scienceqa_dataset = ScienceQADataset()

    data_collator = ScienceQAMultimodalDataCollator(processor)

    dataloader = DataLoader(scienceqa_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

    for batch in dataloader:
        print(batch)
        break