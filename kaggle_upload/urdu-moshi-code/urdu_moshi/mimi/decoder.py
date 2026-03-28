from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from configs.model_config import MimiDecoderConfig
from mimi.encoder import BottleneckTransformer, ResidualBlock


class CausalTransposedConv1d(nn.Module):
    features: int
    kernel_size: int
    stride: int = 1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            dtype=self.dtype,
            padding="CAUSAL",
        )(x)
        trim = self.kernel_size - self.stride
        if trim > 0:
            x = x[:, :-trim, :]
        return x


class DecoderBlock(nn.Module):
    out_channels: int
    stride: int
    num_residual_layers: int = 3
    dilation_growth: int = 2
    kernel_size: int = 3
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        in_channels = x.shape[-1]
        x = nn.elu(x)
        x = CausalTransposedConv1d(
            features=self.out_channels,
            kernel_size=2 * self.stride,
            stride=self.stride,
            dtype=self.dtype,
        )(x)
        for i in range(self.num_residual_layers):
            dilation = self.dilation_growth ** i
            x = ResidualBlock(
                channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dtype=self.dtype,
            )(x, train)
        return x


class MimiDecoder(nn.Module):
    config: MimiDecoderConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        cfg = self.config
        channels = [cfg.base_channels * m for m in cfg.channel_multipliers]

        self.input_proj = nn.Dense(cfg.input_dim, use_bias=False, dtype=self.dtype)

        self.post_quant_transformer = BottleneckTransformer(
            model_dim=cfg.transformer_dim,
            mlp_dim=cfg.transformer_mlp_dim,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
            max_context=cfg.transformer_context,
            layer_scale_init=cfg.layer_scale_init,
            dtype=self.dtype,
        )

        blocks = []
        for i, stride in enumerate(cfg.strides):
            out_ch = channels[min(i + 1, len(channels) - 1)]
            blocks.append(
                DecoderBlock(
                    out_channels=out_ch,
                    stride=stride,
                    num_residual_layers=cfg.num_residual_layers,
                    dilation_growth=cfg.dilation_growth,
                    dtype=self.dtype,
                )
            )
        self.blocks = blocks

        self.output_conv = nn.Conv(
            features=cfg.output_channels,
            kernel_size=(cfg.kernel_size,),
            padding="SAME",
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = self.input_proj(x)
        x = self.post_quant_transformer(x, train)
        for block in self.blocks:
            x = block(x, train)
        x = nn.elu(x)
        x = self.output_conv(x)
        x = x.transpose(0, 2, 1)
        return x
