from typing import Any, Literal, Optional, Union, Tuple

import torch
import torch.optim
from torch.nn import Linear, CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPVisionModel,
    ViltConfig,
    ViltModel,
    ViltMLMHead,
    ViltPretrainedModel,
    PreTrainedModel,
    SchedulerType,
)
from transformers.modeling_outputs import MaskedLMOutput

from . import ViltT, MultimodalModelClass


class ViltForPretrain(ViltPretrainedModel):
    _tied_weights_keys = ["mlm_head.decoder.weight", "mlm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        self.mlm_head = ViltMLMHead(config)
        self.itm_head = Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        itm_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        
        batch_size = input_ids.size(0) // 2  # NOTE: the first half is the matched image-text pairs (for mlm_loss), and the second half is the unmatched (for itm_loss).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        # split up final hidden states into text and image features
        text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        text_features, _ = (sequence_output[:batch_size, :text_seq_len], sequence_output[:batch_size, text_seq_len:])

        mlm_logits = self.mlm_head(text_features)
        itm_logits = self.itm_head(pooled_output)

        # Compute loss
        loss, mlm_loss, itm_loss = None, None, None
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        if mlm_labels is not None:
            # move labels to correct device to enable PP
            mlm_labels = mlm_labels.to(mlm_logits.device)
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        
        if itm_labels is not None:
            # move labels to correct device to enable PP
            itm_labels = itm_labels.to(mlm_logits.device)
            itm_loss = loss_fct(itm_logits.view(-1, 2), itm_labels.view(-1))
        
        if mlm_labels is not None and itm_labels is not None:
            loss = mlm_loss + itm_loss

        if not return_dict:
            output = (mlm_logits, itm_logits,) + outputs[2:]
            return ((loss, mlm_loss, itm_loss,) + output)

        return MaskedLMOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            itm_loss=itm_loss,
            mlm_logits=mlm_logits,
            itm_logits=itm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViltaPretrainModelClass(MultimodalModelClass[ViltT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        # Construct vilt config
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

        vision_config = AutoConfig.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").vision_config
        text_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        config = ViltConfig(
            vocab_size=text_config.vocab_size,
            max_position_embeddings=2048,
            hidden_size=vision_config.hidden_size,
            num_hidden_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            intermediate_size=vision_config.intermediate_size,
            hidden_act=vision_config.hidden_act,
            hidden_dropout_prob=vision_config.dropout,
            image_size=processor.image_processor.crop_size["height"],
            patch_size=vision_config.patch_size,
        )
        model = ViltForPretrain(config)

        # Load pretrained vision model and Initialize ViltModel
        model.vision_tower = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", cache_dir=".cache/huggingface")

        # Sanity check
        num_trainable_params = self.get_num_trainable_params(model)
        print(f"num_trainable_params: {num_trainable_params}", flush=True)

        return model

    def get_num_trainable_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @property
    def supports_activation_checkpointing(self) -> bool:
        """Some models don't implement activation (aka gradient) checkpointing. Override and return False if so.
        Refer to PreTrainedModel.supports_gradient_checkpointing.
        You can also implement it yourself (see convnext.py for an example).
        """
        return True

    @property
    def supports_compilation(self) -> bool:
        """Some models do not support torch.compile. Override and return False if so."""
        return True

    @property
    def batch_size(self) -> int:
        """Overall batch size. In our scripts, (num_nodes * gpus_per_node * micro_batch_size * grad_acc_steps)
        always equals batch_size."""
        return 256

    @property
    def training_steps(self) -> int:
        """Total number of training steps."""
        return 2180

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        """Whether to used mixed precision. None if only fp32 precision."""
        return None

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        """The PyTorch optimizer class (not instantiated object), e.g. `torch.optim.AdamW`."""
        return torch.optim.AdamW

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the optimizer class. Not including `params`."""
        return {
            "lr": 1e-4,
            "weight_decay": 0.01,
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        """Learning rate scheduler, referring to implementations in HuggingFace Transformers.
        transformers.SchedulerType (https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.SchedulerType)"""
        return SchedulerType.LINEAR

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for scheduler. Not including `optimizer` or `num_training_steps`."""
        return {
            "num_warmup_steps": int(self.training_steps * 0.10),
            "num_training_steps": self.training_steps,
        }

    @property
    def max_grad_norm(self) -> float:
        """Maximum gradient norm (for gradient clipping)."""
        return 0.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        """Any extra hyper-parameters for transformers.TrainingArguments."""
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        """Name of modules to wrap as FSDP units. Usually the significant model layers, e.g. `['GPTNeoXLayer']`."""
        return ["ViltLayer"]

    @property
    def image_size(self) -> int:
        return 224

    @property
    def vocab_size(self) -> int:
        return 128256

    @property
    def sequence_length(self) -> int:
        return 2048

