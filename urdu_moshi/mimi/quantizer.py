from typing import Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from configs.model_config import SplitRVQConfig


class VQOutput(NamedTuple):
    quantized: jnp.ndarray
    codes: jnp.ndarray
    commitment_loss: jnp.ndarray
    codebook_loss: jnp.ndarray


class VectorQuantizer(nn.Module):
    codebook_size: int
    dim: int
    commitment_weight: float = 0.25
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=0.02),
            (self.codebook_size, self.dim),
        )
        self.ema_cluster_size = self.variable(
            "ema",
            "cluster_size",
            lambda: jnp.ones(self.codebook_size),
        )
        self.ema_embed_sum = self.variable(
            "ema",
            "embed_sum",
            lambda: nn.initializers.normal(stddev=0.02)(
                jax.random.PRNGKey(0), (self.codebook_size, self.dim)
            ),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = False,
        ema_decay: float = 0.99,
    ) -> VQOutput:
        x_flat = x.reshape(-1, self.dim).astype(jnp.float32)
        codebook = self.codebook.astype(jnp.float32)

        distances = (
            jnp.sum(x_flat ** 2, axis=-1, keepdims=True)
            - 2 * jnp.dot(x_flat, codebook.T)
            + jnp.sum(codebook ** 2, axis=-1)
        )
        codes_flat = jnp.argmin(distances, axis=-1)
        quantized_flat = codebook[codes_flat]

        if train:
            one_hot = jax.nn.one_hot(codes_flat, self.codebook_size)
            new_cluster_size = jnp.sum(one_hot, axis=0)
            new_embed_sum = jnp.dot(one_hot.T, x_flat)

            updated_cluster_size = (
                ema_decay * self.ema_cluster_size.value + (1 - ema_decay) * new_cluster_size
            )
            updated_embed_sum = (
                ema_decay * self.ema_embed_sum.value + (1 - ema_decay) * new_embed_sum
            )

            n = jnp.sum(updated_cluster_size)
            smoothed = (updated_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n

            if not self.is_initializing():
                self.ema_cluster_size.value = updated_cluster_size
                self.ema_embed_sum.value = updated_embed_sum
                new_codebook = updated_embed_sum / smoothed[:, None]
                self.codebook.value = new_codebook.astype(self.codebook.value.dtype)

        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized_flat) - x_flat) ** 2)
        codebook_loss = jnp.mean((quantized_flat - jax.lax.stop_gradient(x_flat)) ** 2)

        quantized_flat_st = x_flat + jax.lax.stop_gradient(quantized_flat - x_flat)
        quantized = quantized_flat_st.reshape(x.shape).astype(self.dtype)
        codes = codes_flat.reshape(x.shape[:-1])

        return VQOutput(quantized, codes, commitment_loss, codebook_loss)


class SplitRVQ(nn.Module):
    config: SplitRVQConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        cfg = self.config

        self.input_proj = nn.Dense(cfg.projected_dim, use_bias=False, dtype=self.dtype)
        self.output_proj = nn.Dense(cfg.latent_dim, use_bias=False, dtype=self.dtype)

        self.semantic_vq = VectorQuantizer(
            codebook_size=cfg.semantic_codebook_size,
            dim=cfg.projected_dim,
            commitment_weight=cfg.commitment_weight,
            dtype=self.dtype,
        )

        self.acoustic_vqs = [
            VectorQuantizer(
                codebook_size=cfg.codebook_size,
                dim=cfg.projected_dim,
                commitment_weight=cfg.commitment_weight,
                dtype=self.dtype,
            )
            for _ in range(cfg.num_acoustic_quantizers)
        ]

        self.distillation_proj = nn.Dense(
            cfg.distillation_proj_dim,
            use_bias=False,
            dtype=jnp.float32,
        )

    def __call__(
        self,
        z: jnp.ndarray,
        train: bool = False,
        apply_quantization: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_proj = self.input_proj(z.astype(self.dtype))

        semantic_out = self.semantic_vq(z_proj, train)
        semantic_tokens = semantic_out.codes
        semantic_quantized = semantic_out.quantized

        distillation_features = self.distillation_proj(
            semantic_quantized.astype(jnp.float32)
        )

        residual = z_proj - jax.lax.stop_gradient(semantic_quantized)
        acoustic_quantized_sum = jnp.zeros_like(z_proj)
        acoustic_tokens_list = []
        total_commitment_loss = semantic_out.commitment_loss
        total_codebook_loss = semantic_out.codebook_loss

        for vq in self.acoustic_vqs:
            aq_out = vq(residual, train)
            acoustic_tokens_list.append(aq_out.codes)
            acoustic_quantized_sum = acoustic_quantized_sum + aq_out.quantized
            residual = residual - jax.lax.stop_gradient(aq_out.quantized)
            total_commitment_loss = total_commitment_loss + aq_out.commitment_loss
            total_codebook_loss = total_codebook_loss + aq_out.codebook_loss

        acoustic_tokens = jnp.stack(acoustic_tokens_list, axis=-1)

        combined = semantic_quantized + acoustic_quantized_sum
        reconstructed = self.output_proj(combined)

        all_tokens = jnp.concatenate(
            [semantic_tokens[..., None], acoustic_tokens], axis=-1
        )

        commitment_loss = self.config.commitment_weight * total_commitment_loss
        return reconstructed, all_tokens, commitment_loss, distillation_features

    def decode_tokens(self, tokens: jnp.ndarray) -> jnp.ndarray:
        semantic_tokens = tokens[..., 0]
        acoustic_tokens = tokens[..., 1:]

        semantic_emb = self.semantic_vq.codebook[semantic_tokens]
        acoustic_emb = jnp.zeros_like(semantic_emb)
        for i, vq in enumerate(self.acoustic_vqs):
            acoustic_emb = acoustic_emb + vq.codebook[acoustic_tokens[..., i]]

        combined = semantic_emb + acoustic_emb
        return self.output_proj(combined.astype(self.dtype))

    def compute_distillation_loss(
        self,
        distillation_features: jnp.ndarray,
        wavlm_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        pred = distillation_features / (
            jnp.linalg.norm(distillation_features, axis=-1, keepdims=True) + 1e-8
        )
        target = wavlm_targets / (
            jnp.linalg.norm(wavlm_targets, axis=-1, keepdims=True) + 1e-8
        )
        loss = jnp.mean(1.0 - jnp.sum(pred * target, axis=-1))
        return loss
