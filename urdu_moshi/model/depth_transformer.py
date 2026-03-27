from typing import Optional, List
import jax
import jax.numpy as jnp
import flax.linen as nn

from configs.model_config import DepthTransformerConfig


class RoPEDepth(nn.Module):
    dim: int
    base: float = 10000.0
    dtype: jnp.dtype = jnp.bfloat16

    def __call__(self, seq_len: int) -> jnp.ndarray:
        half_dim = self.dim // 2
        freqs = 1.0 / (self.base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        angles = jnp.outer(t, freqs)
        cos = jnp.cos(angles).astype(self.dtype)
        sin = jnp.sin(angles).astype(self.dtype)
        return jnp.concatenate([cos, sin], axis=-1)


def _apply_rope_depth(x: jnp.ndarray, rope: jnp.ndarray) -> jnp.ndarray:
    half = rope.shape[-1] // 2
    cos, sin = rope[..., :half], rope[..., half:]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class DepthwiseMHSA(nn.Module):
    model_dim: int
    num_heads: int
    depth_index: int
    num_depth_streams: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        B, K, D = x.shape
        head_dim = D // self.num_heads

        q = nn.Dense(D, use_bias=False, dtype=self.dtype, name="q_proj")(x)
        k = nn.Dense(D, use_bias=False, dtype=self.dtype, name="k_proj")(x)
        v = nn.Dense(D, use_bias=False, dtype=self.dtype, name="v_proj")(x)

        q = q.reshape(B, K, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, K, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, K, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        if mask is not None:
            attn = jnp.where(mask[None, None], attn, jnp.finfo(self.dtype).min)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(self.dtype)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, K, D)
        out = nn.Dense(D, use_bias=False, dtype=self.dtype, name="o_proj")(out)
        return out


class DepthwiseFFN(nn.Module):
    model_dim: int
    mlp_dim: int
    depth_index: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype, name="gate")(x)
        up = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype, name="up")(x)
        down = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, name="down")(
            nn.silu(gate) * up
        )
        return down


class DepthTransformerLayer(nn.Module):
    model_dim: int
    mlp_dim: int
    num_heads: int
    num_depth_streams: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.attn_layers = [
            DepthwiseMHSA(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                depth_index=k,
                num_depth_streams=self.num_depth_streams,
                dtype=self.dtype,
            )
            for k in range(self.num_depth_streams)
        ]
        self.ffn_layers = [
            DepthwiseFFN(
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                depth_index=k,
                dtype=self.dtype,
            )
            for k in range(self.num_depth_streams)
        ]
        self.norms_attn = [
            nn.RMSNorm(dtype=self.dtype)
            for _ in range(self.num_depth_streams)
        ]
        self.norms_ffn = [
            nn.RMSNorm(dtype=self.dtype)
            for _ in range(self.num_depth_streams)
        ]

    def __call__(
        self,
        x: jnp.ndarray,
        causal_mask: jnp.ndarray,
        active_depth: int,
        train: bool = False,
    ) -> jnp.ndarray:
        results = []
        for k in range(active_depth):
            xk = x[:, :k+1, :]
            norm_xk = self.norms_attn[k](xk)
            mask_k = causal_mask[:k+1, :k+1]
            attn_out = self.attn_layers[k](norm_xk, mask_k, train)
            xk = xk + attn_out

            norm_xk = self.norms_ffn[k](xk)
            ffn_out = self.ffn_layers[k](norm_xk)
            xk = xk + ffn_out

            results.append(xk[:, -1:, :])

        return jnp.concatenate(results, axis=1)


class DepthTransformer(nn.Module):
    config: DepthTransformerConfig
    backbone_dim: int = 1536
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        cfg = self.config
        self.context_proj = nn.Dense(cfg.model_dim, use_bias=False, dtype=self.dtype)

        self.stream_embeddings = [
            nn.Embed(num_embeddings=2049, features=cfg.model_dim, dtype=self.dtype)
            for _ in range(cfg.num_codebook_streams)
        ]

        self.layers = [
            DepthTransformerLayer(
                model_dim=cfg.model_dim,
                mlp_dim=cfg.mlp_dim,
                num_heads=cfg.num_heads,
                num_depth_streams=cfg.num_codebook_streams,
                dtype=self.dtype,
            )
            for _ in range(cfg.num_layers)
        ]

        self.output_heads = [
            nn.Dense(
                features=2048 if k > 0 else 32000,
                use_bias=False,
                dtype=self.dtype,
            )
            for k in range(cfg.num_codebook_streams)
        ]

        self.norm = nn.RMSNorm(dtype=self.dtype)

        causal_mask = jnp.tril(jnp.ones((cfg.num_codebook_streams, cfg.num_codebook_streams), dtype=bool))
        self.causal_mask = causal_mask

    def __call__(
        self,
        context_vector: jnp.ndarray,
        prev_tokens: jnp.ndarray,
        audio_vocab_sizes: Optional[List[int]] = None,
        train: bool = False,
    ) -> List[jnp.ndarray]:
        B = context_vector.shape[0]
        K = self.config.num_codebook_streams

        z = self.context_proj(context_vector)
        x = z[:, None, :]

        for k in range(K - 1):
            token_emb = self.stream_embeddings[k](prev_tokens[:, k])
            x = jnp.concatenate([x, token_emb[:, None, :]], axis=1)

        for layer in self.layers:
            x = layer(x, self.causal_mask, active_depth=K, train=train)

        x = self.norm(x)

        logits_list = []
        for k in range(K):
            logits_k = self.output_heads[k](x[:, k, :])
            logits_list.append(logits_k)

        return logits_list
