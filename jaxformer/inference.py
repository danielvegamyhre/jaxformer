#!/usr/bin/env python3
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import tiktoken
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from model import Transformer
from config import TrainingConfig

def main(args: Namespace) -> None:
    rng = jax.random.PRNGKey(0)

    # set up tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    jax.debug.print(f"vocab size: {vocab_size}")

    abs_checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    jax.debug.print(f"restoring checkpoint from directory: {abs_checkpoint_dir}")
    restored_checkpoint = checkpoints.restore_checkpoint(ckpt_dir=abs_checkpoint_dir, target=None)
    state, cfg_dict = restored_checkpoint["state"], restored_checkpoint["training_config"]
    cfg = TrainingConfig(**cfg_dict)

    # configure model architecture
    model = Transformer(
            vocab_size=vocab_size,
            num_layers=cfg.num_layers,
            max_seq_len=cfg.seq_len,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.d_model,
            num_heads=cfg.num_attention_heads,
    )

    # add batch dimension (N,) -> (1,N)
    context = np.array(tokenizer.encode(args.prompt))[:, None]

    for _ in tqdm(range(args.num_tokens)):
        next_token_probs = model.apply({"params": state['params']}, context)
        next_token = jnp.argmax(next_token_probs[:, -1, :], axis=-1)
        # add batch dim and add new token to input context for next iteration
        context = np.hstack((context, next_token[np.newaxis, :]))

    # squeeze out batch dim before decoding
    output_text = tokenizer.decode(context.squeeze().tolist())
    print(f"\noutput tokens: {context}")
    print(f"output text: {output_text}")

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--prompt", type=str)
    argparser.add_argument("--checkpoint-dir", type=str)
    argparser.add_argument("--num-tokens", type=int, default=10)
    args = argparser.parse_args()
    main(args)