from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # hyperparams
    steps: int
    learning_rate: float
    batch_size: int

    # accelerators
    device: str
    mixed_precision: str

    # model dims
    num_layers: int
    embed_dim: int
    d_model: int
    num_attention_heads: int
    seq_len: int
    max_output_tokens: int

    # eval configs
    eval_interval: int
    eval_iters: int

    # checkpointing
    checkpoint_interval: int
    checkpoint_dir: str

    # dataset configs
    dataset_file: str
    dataset_dir: str

    # distributed training
    distributed: bool

    # performance analysis
    estimate_mfu: bool
    hardware_peak_tflops: float

    # observability and debugging
    tensorboard_log_dir: str
    wandb_project: str
    wandb_api_key: str
    plot_learning_curves: bool
    debug: bool