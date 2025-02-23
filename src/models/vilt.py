from typing import Any, Literal, Optional, Union, Tuple

import torch
import torch.optim
from torch.nn import Linear, CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPVisionModel,
    ViltConfig,
    PreTrainedModel,
    SchedulerType,
)
from transformers import ViltModel as HFViltModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.vilt.modeling_vilt import ViltMLMHead, ViltPreTrainedModel

from . import ViltT, MultimodalModelClass
    

class ViltForPretrain(ViltPreTrainedModel):
    _tied_weights_keys = ["mlm_head.decoder.weight", "mlm_head.decoder.bias"]

    def __init__(self, config, target_tasks=["mlm", "itm"]):
        super().__init__(config)

        self.target_tasks = target_tasks

        self.vilt = ViltModel(config)
        self.mlm_head = ViltMLMHead(config)
        self.itm_head = Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()
    
    def infer(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels:  Optional[torch.LongTensor] = None,
        return_dict=False,
    ):
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=return_dict,
        )
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels:  Optional[torch.LongTensor] = None,
        mlm_input_ids: Optional[torch.LongTensor] = None,
        mlm_attention_mask: Optional[torch.FloatTensor] = None,
        mlm_token_type_ids: Optional[torch.LongTensor] = None,
        mlm_pixel_values: Optional[torch.FloatTensor] = None,
        mlm_pixel_mask: Optional[torch.LongTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        itm_input_ids: Optional[torch.LongTensor] = None,
        itm_attention_mask: Optional[torch.FloatTensor] = None,
        itm_token_type_ids: Optional[torch.LongTensor] = None,
        itm_pixel_values: Optional[torch.FloatTensor] = None,
        itm_pixel_mask: Optional[torch.LongTensor] = None,
        itm_labels: Optional[torch.LongTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        
        text_seq_len = input_ids.shape[1]
        loss, mlm_loss, itm_loss = None, None, None
        loss_fct = CrossEntropyLoss()  # -100 index = padding token

        if "mlm" in self.target_tasks:
            outputs = self.infer(
                input_ids=mlm_input_ids,
                attention_mask=mlm_attention_mask,
                token_type_ids=mlm_token_type_ids,
                pixel_values=mlm_pixel_values,
                pixel_mask=mlm_pixel_mask,
            )
            sequence_output = outputs[0]
            text_features = sequence_output[:,:text_seq_len]  # Only look at text sequence
            mlm_logits = self.mlm_head(text_features)
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        if "itm" in self.target_tasks:
            outputs = self.infer(
                input_ids=itm_input_ids,
                attention_mask=itm_attention_mask,
                token_type_ids=itm_token_type_ids,
                pixel_values=itm_pixel_values,
                pixel_mask=itm_pixel_mask,
            )
            pooled_output = outputs[1]
            itm_logits = self.itm_head(pooled_output)
            itm_loss = loss_fct(itm_logits.view(-1, 2), itm_labels.view(-1))

        if mlm_labels is not None and itm_labels is not None:
            loss = mlm_loss + itm_loss

        if not return_dict:
            output = (mlm_logits, itm_logits,) + outputs[2:]
            return ((loss, mlm_loss, itm_loss,) + output)

        return {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "itm_loss": itm_loss,
            "mlm_logits": mlm_logits,
            "itm_logits": itm_logits,
        }


class ViltPretrainModelClass(MultimodalModelClass[ViltT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        # Construct vilt config
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

        vision_config = AutoConfig.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").vision_config
        # text_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        config = ViltConfig(
            vocab_size=self.vocab_size,
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
        model = ViltForPretrain(config, target_tasks=["mlm", "itm"])

        # Load pretrained vision model and Initialize ViltModel
        model.vilt.encoder = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", cache_dir=".cache/huggingface").vision_model.encoder

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
        return False

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
        return ["CLIPEncoderLayer"]

    @property
    def image_size(self) -> int:
        return 224

    @property
    def vocab_size(self) -> int:
        return 128256

    @property
    def sequence_length(self) -> int:
        return 2048


class ViltFinetuneModelClass(MultimodalModelClass[ViltT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        # Construct vilt config
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

        vision_config = AutoConfig.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").vision_config
        # text_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        # Load pretrained checkpoint
        ckpt_path = "/gpfs/data/epavlick/share/vilt-pretrain/checkpoint-2180"
        model = ViltForPretrain.from_pretrained(ckpt_path)
        model.target_tasks = ["mlm"]

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
        return False

    @property
    def supports_compilation(self) -> bool:
        """Some models do not support torch.compile. Override and return False if so."""
        return True

    @property
    def batch_size(self) -> int:
        """Overall batch size. In our scripts, (num_nodes * gpus_per_node * micro_batch_size * grad_acc_steps)
        always equals batch_size."""
        return 128

    @property
    def training_steps(self) -> int:
        """Total number of training steps."""
        return 5197

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
        return ["CLIPEncoderLayer"]

    @property
    def image_size(self) -> int:
        return 224

    @property
    def vocab_size(self) -> int:
        return 128256

    @property
    def sequence_length(self) -> int:
        return 2048


class ViltModel(HFViltModel):
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
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        text_batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
        elif pixel_values is None and image_embeds is None:
            raise ValueError("You have to specify either pixel_values or image_embeds")

        image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
        if image_batch_size != text_batch_size:
            raise ValueError("The text inputs and image inputs need to have the same batch size")
        if pixel_mask is None:
            pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # NOTE: We assume `head_mask` to be 1.0 all the time.

        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )