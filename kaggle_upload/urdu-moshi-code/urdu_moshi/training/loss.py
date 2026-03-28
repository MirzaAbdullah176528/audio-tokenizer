from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp


TOTAL_K = 17
K_TEXT = 0
K_MOSHI_SEMANTIC = 1
K_MOSHI_ACOUSTIC_START = 2
K_MOSHI_ACOUSTIC_END = 8
K_USER_SEMANTIC = 9
K_USER_ACOUSTIC_START = 10
K_USER_ACOUSTIC_END = 16


def cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
    return -jnp.sum(one_hot * log_probs, axis=-1)


def moshi_loss(
    all_logits: List[jnp.ndarray],
    joint_tokens: jnp.ndarray,
    loss_weights: jnp.ndarray,
    semantic_loss_weight: float = 100.0,
    acoustic_loss_weight: float = 1.0,
    text_loss_weight: float = 1.0,
    padding_weight: float = 0.5,
    pad_token_id: int = 0,
    is_multi_stream: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    B, S, K = joint_tokens.shape

    shifted_tokens = joint_tokens[:, 1:, :]
    S_pred = S - 1

    per_stream_losses = []
    per_stream_weights = []

    text_logits = all_logits[K_TEXT][:, :S_pred, :]
    text_targets = shifted_tokens[:, :, K_TEXT]
    text_ce = cross_entropy(text_logits, text_targets)
    text_pad_mask = (text_targets == pad_token_id).astype(jnp.float32)
    text_w = jnp.where(text_pad_mask.astype(bool), padding_weight, text_loss_weight)
    per_stream_losses.append(text_ce)
    per_stream_weights.append(text_w)

    moshi_sem_logits = all_logits[K_MOSHI_SEMANTIC][:, :S_pred, :]
    moshi_sem_targets = shifted_tokens[:, :, K_MOSHI_SEMANTIC]
    moshi_sem_ce = cross_entropy(moshi_sem_logits, moshi_sem_targets)
    per_stream_losses.append(moshi_sem_ce)
    per_stream_weights.append(jnp.full_like(moshi_sem_ce, semantic_loss_weight))

    for k in range(K_MOSHI_ACOUSTIC_START, K_MOSHI_ACOUSTIC_END + 1):
        logits_k = all_logits[k][:, :S_pred, :]
        targets_k = shifted_tokens[:, :, k]
        ce_k = cross_entropy(logits_k, targets_k)
        per_stream_losses.append(ce_k)
        per_stream_weights.append(jnp.full_like(ce_k, acoustic_loss_weight))

    if is_multi_stream:
        user_sem_logits = all_logits[K_USER_SEMANTIC][:, :S_pred, :]
        user_sem_targets = shifted_tokens[:, :, K_USER_SEMANTIC]
        user_sem_ce = cross_entropy(user_sem_logits, user_sem_targets)
        per_stream_losses.append(user_sem_ce)
        per_stream_weights.append(jnp.full_like(user_sem_ce, semantic_loss_weight))

        for k in range(K_USER_ACOUSTIC_START, K_USER_ACOUSTIC_END + 1):
            logits_k = all_logits[k][:, :S_pred, :]
            targets_k = shifted_tokens[:, :, k]
            ce_k = cross_entropy(logits_k, targets_k)
            per_stream_losses.append(ce_k)
            per_stream_weights.append(jnp.full_like(ce_k, acoustic_loss_weight))

    all_losses = jnp.stack(per_stream_losses, axis=-1)
    all_weights = jnp.stack(per_stream_weights, axis=-1)

    text_contrib = jnp.mean(all_losses[:, :, 0] * all_weights[:, :, 0])

    audio_losses = all_losses[:, :, 1:]
    audio_weights = all_weights[:, :, 1:]
    audio_weight_sum = jnp.sum(audio_weights, axis=-1, keepdims=True)
    audio_contrib = jnp.mean(
        jnp.sum(audio_losses * audio_weights, axis=-1) / (audio_weight_sum[..., 0] + 1e-8)
    )

    total_loss = text_contrib + audio_contrib

    per_stream_mean = jnp.mean(all_losses * all_weights, axis=(0, 1))

    return total_loss, per_stream_mean


def mimi_finetune_loss(
    reconstructed_audio: jnp.ndarray,
    target_audio: jnp.ndarray,
    commitment_loss: jnp.ndarray,
    distillation_loss: Optional[jnp.ndarray] = None,
    commitment_weight: float = 1.0,
    distillation_weight: float = 1.0,
    discriminator_feature_loss: Optional[jnp.ndarray] = None,
    feature_loss_weight: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if discriminator_feature_loss is not None:
        reconstruction_loss = discriminator_feature_loss
    else:
        reconstruction_loss = jnp.mean((reconstructed_audio - target_audio) ** 2)

    total = commitment_weight * commitment_loss

    if distillation_loss is not None:
        total = total + distillation_weight * distillation_loss

    total = total + feature_loss_weight * reconstruction_loss

    return total, jnp.stack([reconstruction_loss, commitment_loss])
