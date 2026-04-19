from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MimiEncoderConfig:
    input_channels: int = 1
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8, 16)
    strides: Tuple[int, ...] = (4, 5, 6, 8, 2)
    kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth: int = 2
    num_residual_layers: int = 3
    lstm_layers: int = 0
    transformer_layers: int = 8
    transformer_heads: int = 8
    transformer_dim: int = 512
    transformer_mlp_dim: int = 2048
    transformer_context: int = 250
    layer_scale_init: float = 0.01
    output_dim: int = 512


@dataclass
class MimiDecoderConfig:
    input_dim: int = 512
    base_channels: int = 512
    channel_multipliers: Tuple[int, ...] = (16, 8, 4, 2, 1)
    strides: Tuple[int, ...] = (2, 8, 6, 5, 4)
    kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth: int = 2
    num_residual_layers: int = 3
    transformer_layers: int = 8
    transformer_heads: int = 8
    transformer_dim: int = 512
    transformer_mlp_dim: int = 2048
    transformer_context: int = 250
    layer_scale_init: float = 0.01
    output_channels: int = 1


@dataclass
class SplitRVQConfig:
    latent_dim: int = 512
    projected_dim: int = 256
    num_acoustic_quantizers: int = 7
    codebook_size: int = 2048
    codebook_dim: int = 256
    semantic_codebook_size: int = 2048
    commitment_weight: float = 0.25
    quantizer_dropout: bool = True
    quantization_rate: float = 0.5
    wavlm_dim: int = 1024
    distillation_proj_dim: int = 1024
    frame_rate: float = 12.5


@dataclass
class MimiConfig:
    encoder: MimiEncoderConfig = field(default_factory=MimiEncoderConfig)
    decoder: MimiDecoderConfig = field(default_factory=MimiDecoderConfig)
    quantizer: SplitRVQConfig = field(default_factory=SplitRVQConfig)
    sample_rate: int = 24000
    frame_rate: float = 12.5
    num_quantizers: int = 8
    pretrained_hf_repo: str = "kyutai/moshika-pytorch-bf16"


@dataclass
class DepthTransformerConfig:
    model_dim: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 6
    num_codebook_streams: int = 17
    dropout_rate: float = 0.0
    rope_base: float = 10000.0
    use_depthwise_params: bool = True
    causal: bool = True


@dataclass
class Qwen25BackboneConfig:
    hf_model_id: str ="Qwen/Qwen2.5-0.5B"
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    max_position_embeddings: int = 131072
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True


@dataclass
class AbstractBackboneConfig:
    model_dim: int = 1536
    vocab_size: int = 151936
    max_seq_len: int = 4096


@dataclass
class MultiStreamConfig:
    num_codebooks: int = 8
    frame_rate: float = 12.5
    acoustic_delay_pretrain: int = 2
    acoustic_delay_finetune: int = 1
    num_streams: int = 2
    total_depth_streams: int = 17


@dataclass
class InnerMonologueConfig:
    text_vocab_size: int = 32000
    pad_token_id: int = 0
    epad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    text_delay_pretrain_min: float = -0.6
    text_delay_pretrain_max: float = 0.6
    text_delay_finetune: float = 0.0
    text_mask_prob_pretrain: float = 0.3
    padding_token_weight: float = 0.5


@dataclass
class RQTransformerConfig:
    backbone: Qwen25BackboneConfig = field(default_factory=Qwen25BackboneConfig)
    depth: DepthTransformerConfig = field(default_factory=DepthTransformerConfig)
    multistream: MultiStreamConfig = field(default_factory=MultiStreamConfig)
    inner_monologue: InnerMonologueConfig = field(default_factory=InnerMonologueConfig)
    audio_vocab_size: int = 2048
    dtype: str = "bfloat16"


@dataclass
class MoshiConfig:
    mimi: MimiConfig = field(default_factory=MimiConfig)
    rq_transformer: RQTransformerConfig = field(default_factory=RQTransformerConfig)
    context_steps: int = 3000
    context_duration_seconds: float = 240.0
    semantic_loss_weight: float = 100.0
    acoustic_loss_weight: float = 1.0
    text_loss_weight: float = 1.0
