from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from configs.model_config import MimiEncoderConfig


def make_causal_conv_mask(seq_len: int) -> jnp.ndarray:
    return jnp.ones((seq_len, seq_len), dtype=bool)


class CausalConv1d(nn.Module):
    features: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    groups: int = 1
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pad_total = self.dilation * (self.kernel_size - 1)
        x = jnp.pad(x, ((0, 0), (pad_total, 0), (0, 0)))
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            kernel_dilation=(self.dilation,),
            feature_group_count=self.groups,
            use_bias=self.use_bias,
            dtype=self.dtype,
            padding="VALID",
        )(x)
        return x


class WeightNormConv1d(nn.Module):
    features: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    causal: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.causal:
            return CausalConv1d(
                features=self.features,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                dtype=self.dtype,
            )(x)
        else:
            pad = self.dilation * (self.kernel_size - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)))
            return nn.Conv(
                features=self.features,
                kernel_size=(self.kernel_size,),
                strides=(self.stride,),
                kernel_dilation=(self.dilation,),
                dtype=self.dtype,
                padding="VALID",
            )(x)


class ResidualBlock(nn.Module):
    channels: int
    kernel_size: int = 3
    dilation: int = 1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        residual = x
        h = nn.elu(x)
        h = WeightNormConv1d(
            features=self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            dtype=self.dtype,
        )(h)
        h = nn.elu(h)
        h = WeightNormConv1d(
            features=self.channels,
            kernel_size=1,
            dtype=self.dtype,
        )(h)
        return h + residual


class EncoderBlock(nn.Module):
    out_channels: int
    stride: int
    num_residual_layers: int = 3
    dilation_growth: int = 2
    kernel_size: int = 3
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        in_channels = x.shape[-1]
        for i in range(self.num_residual_layers):
            dilation = self.dilation_growth ** i
            x = ResidualBlock(
                channels=in_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dtype=self.dtype,
            )(x, train)
        x = nn.elu(x)
        x = WeightNormConv1d(
            features=self.out_channels,
            kernel_size=2 * self.stride,
            stride=self.stride,
            dtype=self.dtype,
        )(x)
        return x


class RoPEPositionalEncoding(nn.Module):
    dim: int
    base: float = 10000.0
    dtype: jnp.dtype = jnp.bfloat16

    def __call__(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        half_dim = self.dim // 2
        freqs = 1.0 / (self.base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        angles = jnp.outer(positions, freqs)
        cos = jnp.cos(angles).astype(self.dtype)
        sin = jnp.sin(angles).astype(self.dtype)
        return cos, sin


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class CausalTransformerLayer(nn.Module):
    model_dim: int
    mlp_dim: int
    num_heads: int
    layer_scale_init: float = 0.01
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        B, T, D = x.shape
        head_dim = D // self.num_heads

        qkv = nn.Dense(3 * D, use_bias=False, dtype=self.dtype)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.num_heads, head_dim)
        k = k.reshape(B, T, self.num_heads, head_dim)
        v = v.reshape(B, T, self.num_heads, head_dim)

        cos_exp = cos[:T, :].reshape(1, T, 1, head_dim // 2)
        sin_exp = sin[:T, :].reshape(1, T, 1, head_dim // 2)
        cos_full = jnp.tile(cos_exp, (1, 1, 1, 2))
        sin_full = jnp.tile(sin_exp, (1, 1, 1, 2))

        q = apply_rope(q, cos_full, sin_full)
        k = apply_rope(k, cos_full, sin_full)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn = jnp.where(causal_mask[None, None], attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = nn.Dense(D, use_bias=False, dtype=self.dtype)(out)

        gamma_attn = self.param("ls_gamma_attn", nn.initializers.constant(self.layer_scale_init), (D,))
        x = residual + out * gamma_attn.astype(self.dtype)

        residual2 = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        gate = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype)(x)
        up = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype)(x)
        x = nn.gelu(gate) * up
        x = nn.Dense(D, use_bias=False, dtype=self.dtype)(x)

        gamma_mlp = self.param("ls_gamma_mlp", nn.initializers.constant(self.layer_scale_init), (D,))
        x = residual2 + x * gamma_mlp.astype(self.dtype)

        return x


class BottleneckTransformer(nn.Module):
    model_dim: int
    mlp_dim: int
    num_heads: int
    num_layers: int
    max_context: int
    layer_scale_init: float = 0.01
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.rope = RoPEPositionalEncoding(dim=self.model_dim // self.num_heads, dtype=self.dtype)
        self.layers = [
            CausalTransformerLayer(
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                layer_scale_init=self.layer_scale_init,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm(dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        T = x.shape[1]
        cos, sin = self.rope(T)
        for layer in self.layers:
            x = layer(x, cos, sin, train)
        return self.norm(x)


class MimiEncoder(nn.Module):
    config: MimiEncoderConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        cfg = self.config
        channels = [cfg.base_channels * m for m in cfg.channel_multipliers]

        self.input_conv = WeightNormConv1d(
            features=channels[0],
            kernel_size=cfg.kernel_size,
            dtype=self.dtype,
        )

        blocks = []
        for i, stride in enumerate(cfg.strides):
            out_ch = channels[min(i + 1, len(channels) - 1)]
            blocks.append(
                EncoderBlock(
                    out_channels=out_ch,
                    stride=stride,
                    num_residual_layers=cfg.num_residual_layers,
                    dilation_growth=cfg.dilation_growth,
                    dtype=self.dtype,
                )
            )
        self.blocks = blocks

        self.pre_quant_transformer = BottleneckTransformer(
            model_dim=cfg.transformer_dim,
            mlp_dim=cfg.transformer_mlp_dim,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
            max_context=cfg.transformer_context,
            layer_scale_init=cfg.layer_scale_init,
            dtype=self.dtype,
        )

        self.pre_quant_proj = nn.Dense(cfg.transformer_dim, use_bias=False, dtype=self.dtype)
        self.output_proj = nn.Dense(cfg.output_dim, use_bias=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = x.transpose(0, 2, 1) if x.shape[1] != 1 else x.transpose(0, 2, 1)
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x, train)
        x = nn.elu(x)
        x = self.pre_quant_proj(x)
        x = self.pre_quant_transformer(x, train)
        x = self.output_proj(x)
        return x
