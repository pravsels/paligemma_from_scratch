
import torch 
from torch import nn 
from typing import Optional, Tuple, List 
from torch.nn import CrossEntropyLoss 
import math 
from siglip import SiglipVisionConfig, SiglipVisionTransformer

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

