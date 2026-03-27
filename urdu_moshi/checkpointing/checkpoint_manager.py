from typing import Any, Dict, Optional
import os
import json
from pathlib import Path

import jax
import jax.numpy as jnp

try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        keep_every_n_steps: Optional[int] = 50_000,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.keep_every_n_steps = keep_every_n_steps

        if ORBAX_AVAILABLE:
            options = ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                keep_period=keep_every_n_steps,
                save_interval_steps=1,
            )
            self._manager = ocp.CheckpointManager(
                str(self.checkpoint_dir),
                options=options,
            )
        else:
            self._manager = None
            print("orbax not available, using numpy fallback checkpointing.")

    def save(
        self,
        step: int,
        state: Any,
        stage: int = 1,
        extra_metadata: Optional[Dict] = None,
    ) -> None:
        metadata = {"step": step, "stage": stage}
        if extra_metadata:
            metadata.update(extra_metadata)

        if ORBAX_AVAILABLE:
            self._manager.save(
                step,
                args=ocp.args.StandardSave(state),
            )
        else:
            self._save_numpy(step, state, metadata)

        self._write_metadata(step, metadata)
        print(f"Checkpoint saved at step {step} (stage {stage})")

    def restore(
        self,
        step: Optional[int] = None,
        target: Optional[Any] = None,
    ) -> Optional[Any]:
        if step is None:
            step = self._get_latest_step()
        if step is None:
            print("No checkpoint found.")
            return None

        if ORBAX_AVAILABLE:
            state = self._manager.restore(
                step,
                args=ocp.args.StandardRestore(target) if target is not None else None,
            )
            print(f"Restored checkpoint from step {step}")
            return state
        else:
            return self._load_numpy(step)

    def swap_backbone(
        self,
        current_state: Any,
        new_backbone_params: Dict,
        new_temporal_optimizer_state: Any,
    ) -> Any:
        from flax import struct

        updated_temporal = new_backbone_params
        updated_ema = jax.tree_util.tree_map(
            lambda ema, new: new,
            current_state.ema_params.get("backbone", {}),
            new_backbone_params,
        )
        new_ema = {**current_state.ema_params, "backbone": updated_ema}

        new_state = current_state.replace(
            temporal_params=updated_temporal,
            temporal_opt_state=new_temporal_optimizer_state,
            ema_params=new_ema,
        )

        print("Backbone hot-swapped successfully.")
        print(f"Preserved depth transformer params and step counter (step={current_state.step})")
        return new_state

    def extract_depth_transformer_params(self, state: Any) -> Dict:
        return state.depth_params

    def extract_backbone_params(self, state: Any) -> Dict:
        return state.temporal_params

    def save_component(
        self,
        name: str,
        params: Dict,
        step: int,
    ) -> None:
        component_dir = self.checkpoint_dir / "components" / name
        component_dir.mkdir(parents=True, exist_ok=True)
        save_path = component_dir / f"step_{step}"
        import numpy as np
        flat_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
        np.savez(str(save_path), **{str(i): v for i, v in enumerate(jax.tree_util.tree_leaves(flat_params))})
        print(f"Component '{name}' saved at step {step}")

    def _write_metadata(self, step: int, metadata: Dict) -> None:
        meta_path = self.checkpoint_dir / f"metadata_step_{step}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        latest_path = self.checkpoint_dir / "latest_checkpoint.json"
        with open(latest_path, "w") as f:
            json.dump({"step": step, **metadata}, f, indent=2)

    def _get_latest_step(self) -> Optional[int]:
        latest_path = self.checkpoint_dir / "latest_checkpoint.json"
        if latest_path.exists():
            with open(latest_path) as f:
                data = json.load(f)
            return data.get("step")

        if ORBAX_AVAILABLE and self._manager:
            return self._manager.latest_step()
        return None

    def _save_numpy(self, step: int, state: Any, metadata: Dict) -> None:
        import numpy as np
        save_dir = self.checkpoint_dir / f"step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        leaves, treedef = jax.tree_util.tree_flatten(state)
        np_leaves = [np.array(x) if hasattr(x, "shape") else x for x in leaves]

        for i, (leaf, np_leaf) in enumerate(zip(leaves, np_leaves)):
            if hasattr(np_leaf, "shape"):
                np.save(str(save_dir / f"param_{i}.npy"), np_leaf)

        with open(save_dir / "treedef.json", "w") as f:
            json.dump({"num_leaves": len(leaves)}, f)

    def _load_numpy(self, step: int) -> Optional[Any]:
        import numpy as np
        load_dir = self.checkpoint_dir / f"step_{step}"
        if not load_dir.exists():
            print(f"No numpy checkpoint at step {step}")
            return None

        params = {}
        i = 0
        while (load_dir / f"param_{i}.npy").exists():
            params[i] = jnp.array(np.load(str(load_dir / f"param_{i}.npy")))
            i += 1

        print(f"Loaded {i} parameter arrays from step {step}")
        return params
