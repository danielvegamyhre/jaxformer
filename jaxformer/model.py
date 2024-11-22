import jax
import jax.numpy as jnp
from flax import linen as nn

class Transformer(nn.Module):
    max_seq_len: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # token and positional embeddings
        tok_embed = nn.Embed(self.vocab_size, self.embed_dim)
        pos_embed = nn.Embed(self.max_seq_len, self.embed_dim)

        x = tok_embed(x) + pos_embed(x) # (batch, seq, embed)

        for _ in range(self.num_layers):
            x = DecoderLayer(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim, 
                num_heads=self.num_heads)(x)
        
        x = nn.RMSNorm()(x)
        x = nn.Dense(self.vocab_size)(x)
        x = nn.softmax(x)
        return x

class DecoderLayer(nn.Module):
    embed_dim: int = 128
    hidden_dim: int = 512
    num_heads: int = 4
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # RMS Norm
        activations = nn.RMSNorm()(x)

        # MHA
        activations = nn.attention.MultiHeadAttention(self.num_heads, qkv_features=self.hidden_dim, dtype=jnp.float32)(activations)
        
        # add and norm
        new_activations = nn.RMSNorm()(x + activations)

        # feed forward
        new_activations = FeedForward()(new_activations)

        # residual
        return activations + new_activations

class FeedForward(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        channels = x.shape[-1]
        x = nn.Dense(channels * 4)
        x = nn.relu(x)
        x = nn.Dense(channels)
        return x