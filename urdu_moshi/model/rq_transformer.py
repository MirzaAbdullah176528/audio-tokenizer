from typing import Dict, List, Optional, Tuple, NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from configs.model_config import RQTransformerConfig
from model.backbone import AbstractTemporalBackbone, Qwen25FlaxBackbone
from model.depth_transformer import DepthTransformer


K_TEXT = 0
K_MOSHI_SEMANTIC = 1
K_MOSHI_ACOUSTIC_START = 2
K_MOSHI_ACOUSTIC_END = 8
K_USER_SEMANTIC = 9
K_USER_ACOUSTIC_START = 10
K_USER_ACOUSTIC_END = 16
TOTAL_K = 17


class RQTransformerOutput(NamedTuple):
    all_logits: List[jnp.ndarray]
    context_vectors: jnp.ndarray
    moshi_audio_logits: List[jnp.ndarray]
    user_audio_logits: List[jnp.ndarray]
    text_logits: jnp.ndarray


class RQTransformer(nn.Module):
    config: RQTransformerConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        cfg = self.config

        self.backbone = Qwen25FlaxBackbone(
            config=cfg.backbone,
            dtype=self.dtype,
        )

        self.depth_transformer = DepthTransformer(
            config=cfg.depth,
            backbone_dim=cfg.backbone.hidden_size,
            dtype=self.dtype,
        )

        self.moshi_stream_embeds = [
            nn.Embed(
                num_embeddings=cfg.audio_vocab_size + 1,
                features=cfg.backbone.hidden_size,
                dtype=self.dtype,
            )
            for _ in range(cfg.multistream.num_codebooks)
        ]

        self.user_stream_embeds = [
            nn.Embed(
                num_embeddings=cfg.audio_vocab_size + 1,
                features=cfg.backbone.hidden_size,
                dtype=self.dtype,
            )
            for _ in range(cfg.multistream.num_codebooks)
        ]

        self.text_embed = nn.Embed(
            num_embeddings=cfg.inner_monologue.text_vocab_size + 4,
            features=cfg.backbone.hidden_size,
            dtype=self.dtype,
        )

        self.backbone_to_depth = nn.Dense(
            cfg.depth.model_dim,
            use_bias=False,
            dtype=self.dtype,
        )

    def build_input_embeddings(
        self,
        joint_tokens: jnp.ndarray,
    ) -> jnp.ndarray:
        B, S, K = joint_tokens.shape

        text_tokens = joint_tokens[:, :, K_TEXT]
        text_emb = self.text_embed(text_tokens)

        moshi_embs = []
        for q in range(self.config.multistream.num_codebooks):
            k_idx = K_MOSHI_SEMANTIC + q
            toks = joint_tokens[:, :, k_idx]
            emb = self.moshi_stream_embeds[q](toks)
            moshi_embs.append(emb)

        user_embs = []
        for q in range(self.config.multistream.num_codebooks):
            k_idx = K_USER_SEMANTIC + q
            toks = joint_tokens[:, :, k_idx]
            emb = self.user_stream_embeds[q](toks)
            user_embs.append(emb)

        combined = text_emb
        for emb in moshi_embs:
            combined = combined + emb
        for emb in user_embs:
            combined = combined + emb

        return combined

    def __call__(
        self,
        joint_tokens: jnp.ndarray,
        train: bool = False,
    ) -> RQTransformerOutput:
        B, S, K = joint_tokens.shape

        input_emb = self.build_input_embeddings(joint_tokens)
        context_vectors, backbone_text_logits = self.backbone(input_emb, train)
        depth_context = self.backbone_to_depth(context_vectors)

        all_logits = []
        for s in range(S):
            ctx_s = depth_context[:, s, :]
            prev_k = joint_tokens[:, s, :]
            depth_logits = self.depth_transformer(ctx_s, prev_k, train=train)
            all_logits.append(depth_logits)

        stacked_logits = [
            jnp.stack([all_logits[s][k] for s in range(S)], axis=1)
            for k in range(TOTAL_K)
        ]

        text_logits = stacked_logits[K_TEXT]
        moshi_audio_logits = stacked_logits[K_MOSHI_SEMANTIC:K_MOSHI_ACOUSTIC_END + 1]
        user_audio_logits = stacked_logits[K_USER_SEMANTIC:K_USER_ACOUSTIC_END + 1]

        return RQTransformerOutput(
            all_logits=stacked_logits,
            context_vectors=context_vectors,
            moshi_audio_logits=moshi_audio_logits,
            user_audio_logits=user_audio_logits,
            text_logits=text_logits,
        )

    def sample_step(
        self,
        context_vector: jnp.ndarray,
        prev_tokens_k: jnp.ndarray,
        user_audio_tokens: jnp.ndarray,
        temperature: float = 0.8,
        top_k: int = 50,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        depth_context = self.backbone_to_depth(context_vector)
        B = depth_context.shape[0]

        prev_with_user = jnp.concatenate([prev_tokens_k, user_audio_tokens], axis=-1)
        depth_logits = self.depth_transformer(depth_context, prev_with_user, train=False)

        sampled_text = _sample_logits(depth_logits[K_TEXT], temperature, top_k, key)
        key, subkey = jax.random.split(key) if key is not None else (None, None)

        moshi_audio_samples = []
        for q in range(self.config.multistream.num_codebooks):
            k_idx = K_MOSHI_SEMANTIC + q
            tok = _sample_logits(depth_logits[k_idx], temperature, top_k, subkey)
            moshi_audio_samples.append(tok)
            if subkey is not None:
                key, subkey = jax.random.split(key)

        moshi_audio = jnp.stack(moshi_audio_samples, axis=-1)
        return sampled_text, moshi_audio, key


def _sample_logits(
    logits: jnp.ndarray,
    temperature: float,
    top_k: int,
    key: Optional[jax.random.PRNGKey],
) -> jnp.ndarray:
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_k > 0:
        top_k_vals = jax.lax.top_k(logits, top_k)[0]
        threshold = top_k_vals[:, -1:]
        logits = jnp.where(logits >= threshold, logits, jnp.finfo(logits.dtype).min)

    if key is None:
        key = jax.random.PRNGKey(0)

    return jax.random.categorical(key, logits, axis=-1)


def apply_acoustic_delay(
    tokens: jnp.ndarray,
    delay: int,
) -> jnp.ndarray:
    B, S, Q = tokens.shape
    if delay == 0:
        return tokens

    semantic = tokens[:, :, 0:1]

    acoustic = tokens[:, :, 1:]
    pad = jnp.zeros((B, delay, Q - 1), dtype=acoustic.dtype)
    acoustic_delayed = jnp.concatenate([pad, acoustic[:, :-delay, :]], axis=1)

    return jnp.concatenate([semantic, acoustic_delayed], axis=-1)


def build_joint_sequence(
    moshi_tokens: jnp.ndarray,
    user_tokens: jnp.ndarray,
    text_tokens: jnp.ndarray,
    acoustic_delay: int,
) -> jnp.ndarray:
    moshi_delayed = apply_acoustic_delay(moshi_tokens, acoustic_delay)
    user_delayed = apply_acoustic_delay(user_tokens, acoustic_delay)

    B, S, _ = moshi_delayed.shape
    text_expanded = text_tokens[:, :, None]

    joint = jnp.concatenate([
        text_expanded,
        moshi_delayed,
        user_delayed,
    ], axis=-1)

    return joint
