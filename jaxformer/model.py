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
        batch_size, seq_len = x.shape

        # token and positional embeddings
        pos_indexes = jnp.arange(0, seq_len)
        tok_embed = nn.Embed(self.vocab_size, self.embed_dim)
        pos_embed = nn.Embed(self.max_seq_len, self.embed_dim)
        x = tok_embed(x) + pos_embed(pos_indexes) # (batch, seq, embed)

        # decoder layers
        for _ in range(self.num_layers):
            x = DecoderLayer(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim, 
                num_heads=self.num_heads
            )(x)

        # normalize and project to vocab size
        x = nn.RMSNorm()(x)
        x = nn.Dense(self.vocab_size)(x)

        # loss function optax.softmax_cross_entropy_with_integer_labels
        # expects inputs to be unnormalized log softmax probabilities.
        x = nn.log_softmax(x)
        return x

class DecoderLayer(nn.Module):
    embed_dim: int = 128
    hidden_dim: int = 512
    num_heads: int = 4
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # norm input activations
        normed_inputs = nn.RMSNorm()(x)

        # MHA
        mha_out = nn.attention.MultiHeadAttention(self.num_heads, qkv_features=self.hidden_dim, dtype=jnp.float32)(normed_inputs)
        
        # add residual and norm
        residual_and_norm_out = nn.RMSNorm()(x + mha_out)

        # feed forward
        ffwd_out = FeedForward()(residual_and_norm_out)

        # add residual to ffwd output
        activations = mha_out + ffwd_out
        return activations

class FeedForward(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        channels = x.shape[-1]
        # (batch, seq, channels) @ (channels, 4 * channels) = (batch, seq, 4 * channels)
        x = nn.Dense(channels * 4)(x)
        x = nn.relu(x)
        # (batch, seq, 4 * channels) @ (4 * channels, channels) = (batch, seq, channels)
        x = nn.Dense(channels)(x)
        return x