from typing import Any, Callable, Dict, Optional, Tuple, Union
from functools import partial
import time

import jax
import jax.numpy as jnp
from jax import pmap, lax
import optax
import flax
from flax.training import train_state
from flax import struct

from configs.model_config import MoshiConfig, RQTransformerConfig
from model.rq_transformer import RQTransformer
from training.loss import moshi_loss
from training.scheduler import build_stage1_optimizers, build_finetuning_optimizers
from checkpointing.checkpoint_manager import CheckpointManager


class TrainState(struct.PyTreeNode):
    step: int
    temporal_params: Any
    depth_params: Any
    temporal_opt_state: Any
    depth_opt_state: Any
    ema_params: Any


def partition_params(params: Dict) -> Tuple[Dict, Dict]:
    temporal_params = params.get("backbone", {})
    depth_params = {k: v for k, v in params.items() if k != "backbone"}
    return temporal_params, depth_params


def merge_params(temporal_params: Dict, depth_params: Dict) -> Dict:
    merged = {"backbone": temporal_params}
    merged.update(depth_params)
    return merged


def compute_ema(ema_params: Dict, new_params: Dict, decay: float = 0.9999) -> Dict:
    return jax.tree_util.tree_map(
        lambda ema, new: decay * ema + (1 - decay) * new,
        ema_params,
        new_params,
    )


def create_train_state(
    model: RQTransformer,
    config,
    pretrained_params: Optional[Dict] = None,
    stage: int = 1,
) -> TrainState:
    from model.depth_transformer import DepthTransformer
    from configs.model_config import DepthTransformerConfig

    rng = jax.random.PRNGKey(42)

    depth_cfg = model.config.depth
    backbone_hidden = model.config.backbone.hidden_size

    depth_standalone = DepthTransformer(
        config=depth_cfg,
        backbone_dim=backbone_hidden,
        dtype=jnp.bfloat16,
    )
    dummy_ctx = jnp.zeros((1, backbone_hidden), dtype=jnp.bfloat16)
    dummy_prev = jnp.zeros((1, depth_cfg.num_codebook_streams), dtype=jnp.int32)
    depth_vars = depth_standalone.init(rng, dummy_ctx, dummy_prev)
    depth_init_params = depth_vars["params"]
    del dummy_ctx, dummy_prev, depth_vars
    jax.clear_caches()


def _merge_pretrained_params(init_params: Dict, pretrained: Dict) -> Dict:
    def _merge(init, pre):
        if isinstance(init, dict) and isinstance(pre, dict):
            return {k: _merge(init[k], pre[k]) if k in pre else init[k] for k in init}
        return pre if pre is not None else init
    return _merge(init_params, pretrained)


def make_train_step(
    model: RQTransformer,
    temporal_optimizer: optax.GradientTransformation,
    depth_optimizer: optax.GradientTransformation,
    config,
    is_multi_stream: bool = False,
):
    def loss_fn(params, batch, rng):
        joint_tokens = batch["joint_tokens"]
        is_text_only = batch.get("is_text_only", False)

        all_params = merge_params(params["temporal"], params["depth"])

        output = model.apply(
            {"params": all_params},
            joint_tokens,
            train=True,
            rngs={"dropout": rng},
        )

        total_loss, per_stream_losses = moshi_loss(
            all_logits=output.all_logits,
            joint_tokens=joint_tokens,
            loss_weights=batch.get("loss_weights", None),
            semantic_loss_weight=100.0,
            acoustic_loss_weight=1.0,
            text_loss_weight=1.0,
            padding_weight=0.5,
            is_multi_stream=is_multi_stream and not is_text_only,
        )

        return total_loss, per_stream_losses

    @partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
    def train_step(state: TrainState, batch: Dict, rng: jax.random.PRNGKey) -> Tuple[TrainState, Dict]:
        rng, dropout_rng = jax.random.split(rng)

        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

        combined_params = {"temporal": state.temporal_params, "depth": state.depth_params}
        (loss, per_stream_losses), grads = grad_fn(combined_params, batch, dropout_rng)

        grads = lax.pmean(grads, axis_name="batch")
        loss = lax.pmean(loss, axis_name="batch")

        temporal_grads = grads["temporal"]
        depth_grads = grads["depth"]

        temporal_updates, new_temporal_opt_state = temporal_optimizer.update(
            temporal_grads,
            state.temporal_opt_state,
            state.temporal_params,
        )
        new_temporal_params = optax.apply_updates(state.temporal_params, temporal_updates)

        depth_updates, new_depth_opt_state = depth_optimizer.update(
            depth_grads,
            state.depth_opt_state,
            state.depth_params,
        )
        new_depth_params = optax.apply_updates(state.depth_params, depth_updates)

        new_all_params = merge_params(new_temporal_params, new_depth_params)
        new_ema_params = compute_ema(state.ema_params, new_all_params)

        new_state = TrainState(
            step=state.step + 1,
            temporal_params=new_temporal_params,
            depth_params=new_depth_params,
            temporal_opt_state=new_temporal_opt_state,
            depth_opt_state=new_depth_opt_state,
            ema_params=new_ema_params,
        )

        metrics = {
            "loss": loss,
            "per_stream_losses": per_stream_losses,
        }

        return new_state, metrics

    return train_step


class MoshiTrainer:
    def __init__(
        self,
        model: RQTransformer,
        config,
        stage: int,
        pretrained_params: Optional[Dict] = None,
        checkpoint_dir: str = "/tmp/moshi_checkpoints",
    ):
        self.model = model
        self.config = config
        self.stage = stage
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        temporal_opt, depth_opt = (
            build_stage1_optimizers(config) if stage == 1
            else build_finetuning_optimizers(config)
        )
        self.temporal_optimizer = temporal_opt
        self.depth_optimizer = depth_opt

        is_multi_stream = getattr(config, "use_multistream", False)
        self.train_step = make_train_step(
            model, temporal_opt, depth_opt, config, is_multi_stream
        )

        state = create_train_state(model, config, pretrained_params, stage)
        self.state = jax.device_put_replicated(state, jax.devices())

        num_devices = jax.device_count()
        self.rng = jax.random.split(jax.random.PRNGKey(0), num_devices)

    def train(self, data_iterator, total_steps: Optional[int] = None):
        total_steps = total_steps or self.config.total_steps
        start_step = int(self.state.step[0])

        print(f"Stage {self.stage}: Training from step {start_step} to {total_steps}")
        print(f"Devices: {jax.device_count()} x TPU v5e")

        t0 = time.time()

        for step, batch in enumerate(data_iterator):
            if start_step + step >= total_steps:
                break

            batch = _shard_batch(batch, jax.device_count())

            step_rngs = jax.random.split(self.rng[0], jax.device_count())
            self.rng = jax.random.split(self.rng[0], jax.device_count())

            self.state, metrics = self.train_step(self.state, batch, step_rngs)

            global_step = start_step + step

            if global_step % self.config.log_every_steps == 0:
                loss = float(metrics["loss"][0])
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                print(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss:.4f} | "
                    f"Steps/s: {steps_per_sec:.2f}"
                )

            if global_step % self.config.save_every_steps == 0:
                self._save_checkpoint(global_step)

        self._save_checkpoint(total_steps)
        print(f"Stage {self.stage} training complete.")

    def _save_checkpoint(self, step: int):
        unrep_state = jax.tree_util.tree_map(lambda x: x[0], self.state)
        self.checkpoint_manager.save(
            step=step,
            state=unrep_state,
            stage=self.stage,
        )


def _shard_batch(batch: Dict, num_devices: int) -> Dict:
    def shard(x):
        if hasattr(x, "shape"):
            assert x.shape[0] % num_devices == 0, (
                f"Batch size {x.shape[0]} not divisible by {num_devices} devices"
            )
            return x.reshape((num_devices, x.shape[0] // num_devices) + x.shape[1:])
        return x
    return jax.tree_util.tree_map(shard, batch)


def run_training_stage(
    stage: int,
    model: RQTransformer,
    config,
    data_iterator,
    pretrained_params: Optional[Dict] = None,
    checkpoint_dir: str = "/tmp/moshi_checkpoints",
):
    trainer = MoshiTrainer(
        model=model,
        config=config,
        stage=stage,
        pretrained_params=pretrained_params,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train(data_iterator)
    return trainer.state
