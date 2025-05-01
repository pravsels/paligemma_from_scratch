
import torch 
from torch import nn 
from typing import Optional, Tuple, List 
from torch.nn import CrossEntropyLoss 
import math 
from siglip import SiglipVisionConfig, SiglipVisionTransformer

class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__() 
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,          # index to ignore in targets when computing loss 
        image_token_index=256000,   # index of the <image> token 
        vocab_size=257152,
        projection_dim=2048,        # image patches from ViT get projected to this dim 
        hidden_size=2048,           # embed dims for input (text and image) to the LM
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False 
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemma(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config 
        self.vit = SiglipVisionTransformer(config.vision_config)
        self.vit_projection = PaliGemmaVitProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = Gemma(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_with_image_embeds(
        self, 
        image_embeds: torch.Tensor, 
        input_embeds: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_embeds.shape 
        batch_size, sequence_length = input_ids.shape 
        dtype, device = input_ids.shape 
        # shape: [batch_size, seq_len, hidden_size]
        scaled_image_embeds = image_embeds / (self.config.hidden_size**0.5)

        # combine the embeddings of the image tokens, text tokens and padding tokens 
        # create an embeds of zeros, which will be filled in by the appropriate inputs 
        final_embeds = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)
        # create masks that will help us determine which input to fill in where 
        # shape: [batch_size, seq_len]. mask is true for text tokens 
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # shape: [batch_size, seq_len]. mask is true for image tokens 
        image_mask = input_ids == self.config.image_token_index
        # shape: [batch_size, seq_len]. mask is true for padding 
        pad_mask = input_ids == self.pad_token_id
        
        # add an extra dimension and then expand/broadcast along it to give embed_dim 
        # shape: [batch_size, seq_len, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # add in text embeds from input_embeds. remember that they span the full seq_len and 
        # have <image> tokens as placeholder for image patch embeddings
        final_embeds = torch.where(text_mask_expanded, input_embeds, final_embeds)
        # add in image embeds. we use masked_scatter since scaled image embeds are a subset of total inputs
        final_embeds = final_embeds.masked_scatter(image_mask_expanded, scaled_image_embeds)
        # zero out padding tokens 
        final_embeds = torch.where(pad_mask_expanded, torch.zeros_like(final_embeds), final_embeds)
        
        # Create Attention Mask 
        dtype, device = input_embeds.dtype, input_embeds.device 
        min_dtype = torch.finfo(dtype).min    # most negative float representable 
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0: 
            # fill the mask with 0, which means no masking since we're in the prefill phase 
            # Note: this only works when we have no padding 
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else: 
            # since we're generating tokens, the query is a single token 
            # Note: this only works when we have no padding 
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len

            # here too, we don't mask anything, since the query should be able to attend to all previous input
            # Note: this only works when we have no padding 
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
        
        # add the head dimension 
        # [batch_size, q_len, kv_len] -> [batch_size, num_heads_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)
        
        if kv_cache is not None and kv_cache.num_items() > 0: 
            # query is the last token 
            position_ids = attention_mask.cumsum(-1)[:, -1]
        else: 
            # for masked tokens, we use 1 as position 
            position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask==0), 1).to(device)

        return final_embeds, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,             # input ids to the LM but with <image> tokens 
        pixel_values: torch.FloatTensor = None,         # processed image patches 
        attention_mask: Optional[torch.tensor] = None,  
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple: 
        assert torch.all(attention_mask == 1), "the input cannot be padded"

        # embed input ids 
        # [batch_size, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # embed image patches 
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        image_embeds = self.vit(pixel_values.to(input_embeds.dtype))

        # project image embeds
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        image_embeds = self.vit_projection(image_embeds)

        # image embeds are embedded in place of the <image> tokens 
        input_embeds, attention_mask, position_ids = self._merge_input_with_image_embeds(image_embeds, 
                                                                                         input_embeds, 
                                                                                         input_ids, 
                                                                                         attention_mask, 
                                                                                         kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs 

