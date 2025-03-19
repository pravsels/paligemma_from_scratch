
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
        projection_dim=2048,        # image patches from ViT get projected into 
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


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config 
        self.vit = SiglipVisionTransformer(config.vision_config)
        self.vit_projection = PaliGemmaVitProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 

    def tie_weights(self):
        return self.language_model.tie_weights()
    
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
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_embeds, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs 

