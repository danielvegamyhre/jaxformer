#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import tiktoken

from model import Transformer

BATCH_SIZE = 8
SEQ_LEN = 128

def main() -> None:
    rng = jax.random.PRNGKey(0)

    # set up tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab

    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=128,
        embed_dim=128,
        hidden_dim=512,
        num_heads=4,
    )

    # set up train and validation datasets
    with open('data/shakespeare.txt') as fp:
        data = fp.read()

    train_set_size = int(len(data) * 0.9)
    train_set = data[:train_set_size]
    val_set = data[train_set_size:]

    example_input = jnp.ones((BATCH_SIZE, SEQ_LEN))
    params = model.init(rng, example_input)

    tx = optax.adamw(1e-3)
    train_state = TrainState.create(model.apply, params, tx)

def get_batch(data: str, seq_len: int = 128, batch_size: int = 1):
    ix = jax.random.randint(len(data) - seq_len, (batch_size,))
    x = jnp.stack([data[i:i+seq_len] for i in ix])
    y = jnp.stack([data[i+1:i+1+seq_len] for i in ix])
    return x, y