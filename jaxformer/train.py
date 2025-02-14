#!/usr/bin/env python3
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import tiktoken
import numpy as np
from dataclasses import asdict
from argparse import ArgumentParser

from model import Transformer
from config import TrainingConfig
from data import preprocess

def main(cfg: TrainingConfig) -> None:
    rng = jax.random.PRNGKey(0)

    # set up tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    jax.debug.print(f"vocab size: {vocab_size}")

    # preprocess dataset
    jax.debug.print(f"tokenizing dataset: {cfg.dataset_file}")
    data = preprocess(cfg.dataset_file, tokenizer)

    # set up train and validation datasets
    train_set_size = int(len(data) * 0.9)
    train_set = data[:train_set_size]
    val_set = data[train_set_size:]

    # configure model architecture
    model = Transformer(
            vocab_size=vocab_size,
            num_layers=cfg.num_layers,
            max_seq_len=cfg.seq_len,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.d_model,
            num_heads=cfg.num_attention_heads,
    )

    @jax.jit
    def train_step(state: TrainState, x: jax.Array, y: jax.Array) -> tuple[TrainState, float]:
        """
        `x` has shape (batch size, seq len)
        `y` has shape (batch size, seq len)
        """
        def loss_fn(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
            probs = model.apply({'params': params}, x) # (batch, seq len, vocab size)
            loss = optax.softmax_cross_entropy_with_integer_labels(probs, y).mean()
            return loss
        loss, grads = jax.value_and_grad(loss_fn, allow_int=True)(state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    # train loop
    state = create_train_state(rng, model, cfg)
    for step in range(cfg.steps):
        x, y = get_batch(train_set, seq_len=cfg.seq_len)
        state, loss = train_step(state, x, y) 
        jax.debug.print(f"step: {step} train loss: {loss}")

    # checkpoint
    abs_checkpoint_dir = os.path.abspath(cfg.checkpoint_dir)
    jax.debug.print(f"checkpointing to {abs_checkpoint_dir}")
    checkpoint_data = {"state": state, "training_config": asdict(cfg)}
    checkpoints.save_checkpoint(ckpt_dir=abs_checkpoint_dir, target=checkpoint_data, step=step)

def create_train_state(rng: jax.Array, model: nn.Module, cfg: TrainingConfig) -> TrainState:
    example_input = jnp.ones((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)
    params = model.init(rng, example_input)['params']
    adamw_opt = optax.adamw(cfg.learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=adamw_opt)
    return train_state
    
def get_batch(data: jax.Array, seq_len: int = 128, batch_size: int = 1):
    ix = np.random.randint(0, len(data) - seq_len, (batch_size,))
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+1+seq_len] for i in ix])
    return x, y

if __name__ == '__main__':
    argparser = ArgumentParser()
    # hyperparams
    argparser.add_argument("--steps", type=int, default=100)
    argparser.add_argument("--learning-rate", type=float, default=1e-3)
    argparser.add_argument("--batch-size", type=int, default=32) 

    # acceleration
    argparser.add_argument("--mixed-precision", help="fp16, bfloat16")
    argparser.add_argument("--device", type=str, default="cpu")

    # model dims
    argparser.add_argument("--num-layers", type=int, default=6)
    argparser.add_argument("--embed-dim", type=int, default=128)
    argparser.add_argument("--d-model", type=int, default=128)
    argparser.add_argument("--num-attention-heads", type=int, default=2)
    argparser.add_argument("--seq-len", type=int, default=128)
    argparser.add_argument("--max-output-tokens", type=int, default=128)

    # evaluation
    argparser.add_argument("--eval-interval", type=int, default=100)
    argparser.add_argument("--eval-iters", type=int, default=10)

    # checkpointing
    argparser.add_argument("--checkpoint-interval", type=int, default=100)  
    argparser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # dataset
    argparser.add_argument("--dataset-file", type=str)
    argparser.add_argument("--dataset-dir", type=str)

    # observability
    argparser.add_argument("--tensorboard-log-dir", type=str)
    argparser.add_argument("--wandb-project", type=str)
    argparser.add_argument("--wandb-api-key", type=str)
    argparser.add_argument("--plot-learning-curves", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    # distributed training
    argparser.add_argument('--distributed', action='store_true', help="multi-GPU or multi-node training with distributed data parallelism")

    # performance analysis
    argparser.add_argument("--estimate-mfu", action="store_true", help="estimate MFU then exit")
    argparser.add_argument("--hardware-peak-tflops", type=float, help="Theoretical peak TFLOPs per second achievable on the hardware.")

    args = argparser.parse_args()

    cfg = TrainingConfig(
        # hyperparams
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,

        # acceleration
        device=args.device,
        mixed_precision=args.mixed_precision,

        # model dims
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        seq_len=args.seq_len,
        max_output_tokens=args.max_output_tokens,

        # evaluation
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,

        # checkpointing
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,

        # dataset
        dataset_file=args.dataset_file,
        dataset_dir=args.dataset_dir,

        # distributed training
        distributed=args.distributed,

        # performance analysis
        estimate_mfu=args.estimate_mfu,
        hardware_peak_tflops=args.hardware_peak_tflops,

        # observability and debugging
        plot_learning_curves=args.plot_learning_curves,
        tensorboard_log_dir=args.tensorboard_log_dir,
        wandb_project=args.wandb_project,
        wandb_api_key=args.wandb_api_key,
        debug=args.debug,
    )
    main(cfg)