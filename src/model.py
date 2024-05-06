import torch
import torch.nn as nn
import dataclasses
import typing as tp

from abc import ABC, abstractmethod
from transformers import AutoConfig, LlamaForCausalLM

from src.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from src.brain_encoder import build_brain_encoder
from src.prefix_projector import build_prefix_projector
from src.config import update_config

from bm.events import Event
from bm.studies import Recording


@dataclasses.dataclass
class SegmentBatch:

    meg: torch.Tensor
    subject_index: torch.Tensor
    _recordings: tp.List[Recording] = dataclasses.field(default_factory=list)
    _event_lists: tp.List[tp.List[Event]] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.meg)

class BrainAdapter(ABC):

    def __init__(self, config):
        super(BrainAdapter, self).__init__()
        self.encoder = build_brain_encoder(config)
        self.projector = build_prefix_projector(config)
    
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, eegs, subject_index):
        if self.config.ablations.random_features:
            eegs = torch.rand(eegs.shape, requires_grad=True, device=eegs.device)
        batch = SegmentBatch(meg=eegs, subject_index=subject_index)
        eeg_features = self.encoder(dict(meg=eegs), batch)
        eeg_features = self.projector(eeg_features)
        return eeg_features
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, position_ids, past_key_values, labels, eegs, subject_index):

        image_features = self.encode_images(eegs, subject_index)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        

        image_token_indices = (input_ids == IMAGE_TOKEN_INDEX).nonzero()
        if image_token_indices.size(0) > 0:
            first_image_token_index = image_token_indices[0][1].item()
            prompt_len = first_image_token_index
        else:
            prompt_len = input_ids.shape[1]

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_max_length = getattr(self.config.tokenizer, 'model_max_length', None)
        if tokenizer_max_length is not None:
            new_input_embeds = [x[:tokenizer_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if self.config.tokenizer.padding_side == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            if self.config.tokenizer.ignore_prompt:
                new_labels[:, :prompt_len] = IGNORE_INDEX

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()
    
    def forward(self, x):
        return x
    
class EEGAdapterLlamaForCausalLM(LlamaForCausalLM, BrainAdapter):
    def __init__(self, custom_config, model_name, token):
        config = AutoConfig.from_pretrained(model_name, token=token)
        update_config(config, vars(custom_config.data))
        super(LlamaForCausalLM, self).__init__(config)
        
        self.config = config
        self.encoder = build_brain_encoder(config)
        self.projector = build_prefix_projector(config)

        self.model = LlamaForCausalLM.from_pretrained(model_name, token=token, config=config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = IdentityModule()
        #self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.freeze_weights()
    
    def get_model(self):
        return self.model
    
    def freeze_weights(self):
        param_names = [name for name, cfg in [('model', self.config.llama.freeze), ('encoder', self.config.encoder.freeze)] if cfg]
        for param_name in param_names:
            sub_model = getattr(self, param_name, None)
            if sub_model is not None:
                for param in sub_model.parameters():
                    param.requires_grad = False
            else:
                print(f"No attribute named {param_name} found in the model.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        eegs=None,
        subject_index=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                eegs,
                subject_index,
            )
        # print(f'inputs_embeds: {inputs_embeds.shape}')
        # print(f'labels: {labels}')
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
