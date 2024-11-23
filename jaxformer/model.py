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
        # TODO: migrate to RoPE embeddings
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
        """`x` should have shape (batch, seq, embed)"""
        batch_size, seq_len, embed_dim = x.shape

        # norm input activations
        normed_inputs = nn.RMSNorm()(x)

        # MHA
        # TODO: migrate to GQA with RoPE
        head_dim = self.hidden_dim // self.num_heads

        Q = self.param(
            'Q',
            nn.initializers.normal(),
            (self.embed_dim, self.num_heads, head_dim),
        )

        # (batch, seq, embed) @ (embed, num_heads, head dim) = (batch, seq, num_heads, head_dim)
        q_proj = jnp.einsum("bse,ehd->bshd", x, Q)

        K = self.param(
            'K',
            nn.initializers.normal(),
            (self.embed_dim, self.num_heads, head_dim),
        )

        # (batch, seq, embed) @ (embed, num_heads, head dim) = (batch, seq, num_heads, head_dim)
        k_proj = jnp.einsum("bse,ehd->bshd", x, K)

        # scaled dot product attention
        scale = 1 / Q.shape[-1] ** 0.5
        attention_scores = jnp.einsum("bshd,bthd->bhst", q_proj, k_proj)  # (batch, heads, seq, seq)
        attention_scores *= scale
        causal_mask = nn.make_causal_mask(jax.core.ShapedArray((batch_size, seq_len), dtype=jnp.float32))
        masked_attention_scores = jnp.where(causal_mask > 0, attention_scores, -float('inf'))
        masked_attention_scores = nn.softmax(attention_scores, axis=-1)

        V = self.param(
            'V',
            nn.initializers.normal(),
            (self.embed_dim, self.num_heads, head_dim),
        )
        # (batch, seq, embed) @ (embed, heads, head dim) = (batch, seq, heads, head_dim)
        v_proj = jnp.einsum("bse,ehd->bshd", x, V)

        # (batch, seq, heads, head_dim) @ (batch, heads, seq, seq) = (batch, seq, heads, head dim)
        # (b, h, d, s) @ (b, h, s, s) = (b, h, d, s) -> rearrange -> (b, s, h, d)
        mha_out = jnp.einsum("bshd,bhst->bthd", v_proj, masked_attention_scores)

        # "concat" heads by reshaping
        # (batch, seq, heads, embed)
        mha_out = mha_out.reshape(batch_size, seq_len, self.hidden_dim)

        # output projection from hidden dim back to embed dim to enable residual
        O = self.param(
            'O',
            nn.initializers.normal(),
            (self.hidden_dim, self.embed_dim),
        )
        # (batch, seq, hidden) @ (hidden, embed) = (batch, seq, embed)
        mha_out_proj = jnp.einsum("btd,de->bte", mha_out, O)

        # add residual and norm
        residual_and_norm_out = nn.RMSNorm()(x + mha_out_proj)

        # feed forward
        ffwd_out = FeedForward()(residual_and_norm_out)

        # add residual to ffwd output
        activations = mha_out_proj + ffwd_out
        return activations

class FeedForward(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        channels = x.shape[-1]
        # (batch, seq, channels) @ (channels, 4 * channels) = (batch, seq, 4 * channels)
        x = nn.Dense(channels * 4)(x)
        # TODO: migrate to SwiGLU
        x = nn.relu(x)
        # (batch, seq, 4 * channels) @ (4 * channels, channels) = (batch, seq, channels)
        x = nn.Dense(channels)(x)
        return x