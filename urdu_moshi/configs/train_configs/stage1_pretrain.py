from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stage1TrainConfig:
    stage_name: str = "single_stream_pretrain"

    batch_audio_hours: float = 16.0
    sequence_duration_seconds: float = 300.0
    total_steps: int = 1_000_000
    text_batch_fraction: float = 0.5

    temporal_lr: float = 3e-5
    depth_lr: float = 2e-4
    temporal_text_lr_multiplier: float = 0.75
    lr_warmup_steps: int = 5000
    lr_schedule: str = "cosine"

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0

    acoustic_delay: int = 2
    text_delay_min: float = -0.6
    text_delay_max: float = 0.6
    text_mask_prob: float = 0.3

    use_multistream: bool = False
    semantic_loss_weight: float = 100.0
    padding_loss_weight: float = 0.5

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"

    log_every_steps: int = 100
    save_every_steps: int = 10_000
    eval_every_steps: int = 5_000

    data_path: str = "gs://your-bucket/urdu_audio/stage1"
    checkpoint_dir: str = "gs://your-bucket/checkpoints/stage1"
    tensorboard_dir: str = "gs://your-bucket/tensorboard/stage1"
