import optax
import jax.numpy as jnp
from typing import Optional


def linear_warmup_cosine_decay(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> optax.Schedule:
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    decay = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr_ratio,
    )
    return optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps],
    )


def constant_schedule(lr: float) -> optax.Schedule:
    return optax.constant_schedule(lr)


def linear_warmup_constant(
    base_lr: float,
    warmup_steps: int,
) -> optax.Schedule:
    return optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )


def build_optimizer(
    temporal_lr: float,
    depth_lr: float,
    total_steps: int,
    warmup_steps: int,
    weight_decay: float = 0.1,
    gradient_clip: float = 1.0,
    schedule_type: str = "cosine",
    temporal_text_lr_multiplier: float = 1.0,
) -> optax.GradientTransformation:
    if schedule_type == "cosine":
        temporal_schedule = linear_warmup_cosine_decay(temporal_lr, total_steps, warmup_steps)
        depth_schedule = linear_warmup_cosine_decay(depth_lr, total_steps, warmup_steps)
    else:
        temporal_schedule = constant_schedule(temporal_lr)
        depth_schedule = constant_schedule(depth_lr)

    temporal_optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adamw(
            learning_rate=temporal_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
            mask=lambda p: jnp.ndim(p) > 1,
        ),
    )

    depth_optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adamw(
            learning_rate=depth_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
            mask=lambda p: jnp.ndim(p) > 1,
        ),
    )

    return temporal_optimizer, depth_optimizer


def build_stage1_optimizers(config) -> tuple:
    return build_optimizer(
        temporal_lr=config.temporal_lr,
        depth_lr=config.depth_lr,
        total_steps=config.total_steps,
        warmup_steps=getattr(config, "lr_warmup_steps", 5000),
        weight_decay=config.adamw_weight_decay,
        gradient_clip=config.gradient_clip_norm,
        schedule_type=config.lr_schedule,
    )


def build_finetuning_optimizers(config) -> tuple:
    return build_optimizer(
        temporal_lr=config.temporal_lr,
        depth_lr=config.depth_lr,
        total_steps=config.total_steps,
        warmup_steps=0,
        weight_decay=config.adamw_weight_decay,
        gradient_clip=config.gradient_clip_norm,
        schedule_type="constant",
    )
