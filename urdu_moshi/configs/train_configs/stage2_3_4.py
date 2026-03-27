from dataclasses import dataclass


@dataclass
class Stage2TrainConfig:
    stage_name: str = "multistream_posttraining"

    batch_audio_hours: float = 8.0
    sequence_duration_seconds: float = 300.0
    total_steps: int = 100_000
    text_batch_fraction: float = 0.1

    temporal_lr: float = 3e-6
    depth_lr: float = 5e-5
    lr_schedule: str = "constant"

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0

    acoustic_delay: int = 2
    text_delay: float = 0.0
    text_mask_prob: float = 0.0

    use_multistream: bool = True
    use_diarization: bool = True
    semantic_loss_weight: float = 100.0
    padding_loss_weight: float = 0.5

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"

    log_every_steps: int = 100
    save_every_steps: int = 5_000
    eval_every_steps: int = 2_500

    data_path: str = "gs://your-bucket/urdu_audio/stage2"
    checkpoint_dir: str = "gs://your-bucket/checkpoints/stage2"
    init_from_checkpoint: str = "gs://your-bucket/checkpoints/stage1/best"
    tensorboard_dir: str = "gs://your-bucket/tensorboard/stage2"


@dataclass
class Stage3TrainConfig:
    stage_name: str = "conversation_finetune"

    batch_audio_minutes: float = 40.0
    sequence_duration_seconds: float = 300.0
    total_steps: int = 10_000
    text_batch_fraction: float = 0.0

    temporal_lr: float = 2e-6
    depth_lr: float = 4e-6
    lr_schedule: str = "constant"

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0

    acoustic_delay: int = 1
    text_delay: float = 0.0

    use_multistream: bool = True
    use_real_conversation_data: bool = True
    semantic_loss_weight: float = 100.0
    padding_loss_weight: float = 0.5

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"

    log_every_steps: int = 50
    save_every_steps: int = 1_000
    eval_every_steps: int = 500

    data_path: str = "gs://your-bucket/urdu_audio/stage3_conversations"
    checkpoint_dir: str = "gs://your-bucket/checkpoints/stage3"
    init_from_checkpoint: str = "gs://your-bucket/checkpoints/stage2/best"
    tensorboard_dir: str = "gs://your-bucket/tensorboard/stage3"


@dataclass
class Stage4TrainConfig:
    stage_name: str = "urdu_instruct_finetune"

    batch_audio_hours: float = 2.7
    sequence_duration_seconds: float = 300.0
    total_steps: int = 30_000
    text_batch_fraction: float = 0.0

    temporal_lr: float = 2e-6
    depth_lr: float = 2e-6
    lr_schedule: str = "constant"

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0

    acoustic_delay: int = 1
    text_delay: float = 0.0

    use_multistream: bool = True
    use_synthetic_data: bool = True
    semantic_loss_weight: float = 100.0
    padding_loss_weight: float = 0.5

    user_stream_gain_min_db: float = -24.0
    user_stream_gain_max_db: float = 15.0
    user_stream_gain_prob: float = 0.5
    noise_add_prob: float = 0.3
    noise_snr_min_db: float = -30.0
    noise_snr_max_db: float = 6.0
    echo_scale_min: float = 0.0
    echo_scale_max: float = 0.2
    echo_delay_min_ms: float = 100.0
    echo_delay_max_ms: float = 500.0
    echo_reverb_prob: float = 0.3

    num_tpu_devices: int = 8
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"

    log_every_steps: int = 100
    save_every_steps: int = 2_000
    eval_every_steps: int = 1_000

    data_path: str = "gs://your-bucket/urdu_audio/stage4_instruct"
    checkpoint_dir: str = "gs://your-bucket/checkpoints/stage4"
    init_from_checkpoint: str = "gs://your-bucket/checkpoints/stage3/best"
    tensorboard_dir: str = "gs://your-bucket/tensorboard/stage4"
