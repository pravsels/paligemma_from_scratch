
from typing import Optional, Tuple
import torch 
import torch.nn as nn 

class SiglipVisionConfig:

    def __init__(
        self,
        embed_dim=768,              # embedding dimension of the LM 
        intermediate_size=3072,     # embedding dimension of the MLP layers 
        num_hidden_layers=12,       # no of transformer blocks (each block has attn, mlp and layer norms)
        num_attention_heads=12,     # no of attention heads for each transformer block 
        num_channels=3,             # because image has RGB channels 
        image_size=224,
        patch_size=16,              # image will be broken down into (patch_size, patch_size) blocks
        layer_norm_eps=1e-6,        # epsilon value in the denom of layer norm, to avoid division by zero 
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):

        self.embed_dim = embed_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipImageEmbeddingLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.embed_dim = config.embed_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # conv layer to break down image into (patch_size, patch_size) sized patches 
        self.patch_embedding_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"         # no padding is added 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,      # not saved in the model state dict 
        )


class SiglipTransformerEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layers = nn.ModuleList(
            [SiglipTransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]

        for layer in self.layers:
            embeddings = layer(embeddings)

        return embeddings 

class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        embed_dim = config.embed_dim

        self.embedding_layer = SiglipImageEmbeddingLayer(config)
        self.transformer_encoder = SiglipTransformerEncoder(config)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor: 
        # pixels: [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        embeddings = self.embedding_layer(pixels)

        embeddings = self.transformer_encoder(input_embeds=embeddings)

        embeddings = self.layer_norm(embeddings)

        return embeddings


