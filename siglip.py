
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

    def forward(self, pixels: torch.FloatTensor) -> torch.Tensor: 
        _, _, height, width = pixels.shape  # [batch_size, channels, height, width]

        patch_embeds = self.patch_embedding_conv(pixels)    # [batch_size, embed_dim, num_patches_h, num_patches_w]

        # flattens from dim 2 onwards 
        embeddings = patch_embeds.flatten(2)      # [batch_size, embed_dim, num_patches]; num_patches = num_patches_h * num_patches_w
        # num_patches_h = height // patch_size, num_patches_w = width // patch_size

        embeddings = embeddings.transpose(1, 2)   # [batch_size, num_patches, embed_dim]
        
        # each position encoding is a vector of size embed_dim 
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)

        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):
    """ Multi-headed attention  """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # calculate attention using the formula Q * K^T / sqrt(d_k)
        # atten_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale) 

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the softmax across the row 
        # [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply dropout during training 
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # multiply attn_weights by the value states 
        # [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return attn_output, attn_weights

class SiglipTransformerEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attn_layer = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp_layer = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.attn_layer(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp_layer(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
    
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
            embeddings = layer(input_embeds)

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


