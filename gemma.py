
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

