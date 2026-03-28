"""
Kaggle-specific training configs.
These inherit from the original stage configs and only override
the paths, batch sizes, and save frequencies appropriate for Kaggle's
TPU v5e-8 environment (20GB disk, 12h sessions, 20h/week quota).
"""
from dataclasses import dataclass
from configs.train_configs.stage1_pretrain import Stage1TrainConfig
from configs.train_configs.stage2_3_4 import (
    Stage2TrainConfig,
    Stage3TrainConfig,
    Stage4TrainConfig,
)


@dataclass
class KaggleStage1Config(Stage1TrainConfig):
    data_path: str = "/kaggle/input/urdu-speech-corpus/audio"
    checkpoint_dir: str = "/kaggle/working/checkpoints/stage1"
    tensorboard_dir: str = "/kaggle/working/tensorboard/stage1"

    total_steps: int = 50_000
    save_every_steps: int = 2_000
    log_every_steps: int = 50
    eval_every_steps: int = 2_000

    batch_audio_hours: float = 4.0
    sequence_duration_seconds: float = 120.0

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"


@dataclass
class KaggleStage2Config(Stage2TrainConfig):
    data_path: str = "/kaggle/input/urdu-speech-corpus/diarized_audio"
    checkpoint_dir: str = "/kaggle/working/checkpoints/stage2"
    init_from_checkpoint: str = "/kaggle/input/urdu-moshi-stage1/checkpoints/stage1/best"
    tensorboard_dir: str = "/kaggle/working/tensorboard/stage2"

    total_steps: int = 20_000
    save_every_steps: int = 1_000
    log_every_steps: int = 50

    batch_audio_hours: float = 2.0
    sequence_duration_seconds: float = 120.0

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True


@dataclass
class KaggleStage3Config(Stage3TrainConfig):
    data_path: str = "/kaggle/input/urdu-conversation-corpus/audio"
    checkpoint_dir: str = "/kaggle/working/checkpoints/stage3"
    init_from_checkpoint: str = "/kaggle/input/urdu-moshi-stage2/checkpoints/stage2/best"
    tensorboard_dir: str = "/kaggle/working/tensorboard/stage3"

    total_steps: int = 5_000
    save_every_steps: int = 500
    log_every_steps: int = 25

    batch_audio_minutes: float = 20.0
    sequence_duration_seconds: float = 120.0

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True


@dataclass
class KaggleStage4Config(Stage4TrainConfig):
    data_path: str = "/kaggle/input/urdu-instruct-corpus/audio"
    checkpoint_dir: str = "/kaggle/working/checkpoints/stage4"
    init_from_checkpoint: str = "/kaggle/input/urdu-moshi-stage3/checkpoints/stage3/best"
    tensorboard_dir: str = "/kaggle/working/tensorboard/stage4"

    total_steps: int = 10_000
    save_every_steps: int = 1_000
    log_every_steps: int = 50

    batch_audio_hours: float = 1.0
    sequence_duration_seconds: float = 120.0

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True
