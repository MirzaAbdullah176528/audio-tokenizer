"""
Kaggle-specific checkpoint manager.
Extends the base CheckpointManager to handle:
  - 20GB /kaggle/working disk limit (auto-prune old checkpoints)
  - Session resume from previous Kaggle Dataset commits
  - Best checkpoint tracking for cross-session continuation
"""
import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from checkpointing.checkpoint_manager import CheckpointManager


class KaggleCheckpointManager(CheckpointManager):
    def __init__(
        self,
        checkpoint_dir: str = "/kaggle/working/checkpoints",
        max_to_keep: int = 2,
        best_checkpoint_metric: str = "loss",
        disk_warn_gb: float = 3.0,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            max_to_keep=max_to_keep,
        )
        self.best_checkpoint_metric = best_checkpoint_metric
        self.disk_warn_gb = disk_warn_gb
        self._best_metric_value = float("inf")
        self._best_step = None
        self._saved_steps = []

    def save(
        self,
        step: int,
        state: Any,
        stage: int = 1,
        metrics: Optional[Dict] = None,
        extra_metadata: Optional[Dict] = None,
    ) -> None:
        self._check_disk_space()

        super().save(step=step, state=state, stage=stage, extra_metadata=extra_metadata)
        self._saved_steps.append(step)

        if metrics and self.best_checkpoint_metric in metrics:
            current_metric = metrics[self.best_checkpoint_metric]
            if current_metric < self._best_metric_value:
                self._best_metric_value = current_metric
                self._best_step = step
                self._save_best_pointer(step)
                print(f"  New best checkpoint at step {step} ({self.best_checkpoint_metric}={current_metric:.4f})")

        self._prune_old_checkpoints()

    def _save_best_pointer(self, step: int) -> None:
        best_path = Path(self.checkpoint_dir) / "best_checkpoint.json"
        with open(best_path, "w") as f:
            json.dump({
                "step": step,
                "metric": self.best_checkpoint_metric,
                "value": self._best_metric_value,
            }, f)

    def get_best_step(self) -> Optional[int]:
        best_path = Path(self.checkpoint_dir) / "best_checkpoint.json"
        if best_path.exists():
            with open(best_path) as f:
                return json.load(f)["step"]
        return None

    def restore_best(self, target: Optional[Any] = None) -> Optional[Any]:
        best_step = self.get_best_step()
        if best_step is None:
            print("No best checkpoint found. Restoring latest.")
            return self.restore(target=target)
        print(f"Restoring best checkpoint from step {best_step}")
        return self.restore(step=best_step, target=target)

    def _prune_old_checkpoints(self) -> None:
        if len(self._saved_steps) <= self.max_to_keep:
            return

        steps_to_prune = self._saved_steps[:-self.max_to_keep]
        for step in steps_to_prune:
            if step == self._best_step:
                continue
            step_dir = Path(self.checkpoint_dir) / f"step_{step}"
            if step_dir.exists():
                shutil.rmtree(step_dir)
                print(f"  Pruned old checkpoint at step {step}")
            self._saved_steps.remove(step)

    def _check_disk_space(self) -> None:
        total, used, free = shutil.disk_usage("/kaggle/working")
        free_gb = free / (1024 ** 3)
        if free_gb < self.disk_warn_gb:
            print(
                f"WARNING: Only {free_gb:.1f}GB free in /kaggle/working. "
                f"Pruning aggressively to avoid out-of-disk errors."
            )
            self.max_to_keep = 1
            self._prune_old_checkpoints()

    @staticmethod
    def print_commit_instructions(stage: int) -> None:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║       SAVE YOUR PROGRESS BEFORE SESSION ENDS                 ║
╠══════════════════════════════════════════════════════════════╣
║  Stage {stage} complete. To preserve checkpoints:             ║
║                                                              ║
║  Option A - Kaggle UI:                                       ║
║    Notebook → Save & Run → "Save working directory           ║
║    as a Dataset" → name it "urdu-moshi-stage{stage}"          ║
║                                                              ║
║  Option B - Kaggle API (from your local terminal):           ║
║    kaggle datasets create -p /kaggle/working                 ║
║    --dir-mode zip                                            ║
║                                                              ║
║  Then in the next session, add that dataset as input         ║
║  and set init_from_checkpoint to its path.                   ║
╚══════════════════════════════════════════════════════════════╝
""")

    @staticmethod
    def load_from_previous_session(
        previous_dataset_path: str,
        working_dir: str = "/kaggle/working/checkpoints",
    ) -> Optional[int]:
        """
        Copy checkpoints from a previously committed Kaggle Dataset
        (mounted at /kaggle/input/<dataset-name>/) into /kaggle/working.
        Returns the step number of the latest restored checkpoint.
        """
        src = Path(previous_dataset_path)
        dst = Path(working_dir)
        dst.mkdir(parents=True, exist_ok=True)

        latest_json = src / "latest_checkpoint.json"
        if not latest_json.exists():
            print(f"No latest_checkpoint.json found at {src}")
            return None

        with open(latest_json) as f:
            meta = json.load(f)
        step = meta["step"]

        step_dir = src / f"step_{step}"
        if step_dir.exists():
            dst_step = dst / f"step_{step}"
            if not dst_step.exists():
                shutil.copytree(str(step_dir), str(dst_step))
                print(f"Copied checkpoint step {step} from previous session.")

        shutil.copy(str(latest_json), str(dst / "latest_checkpoint.json"))

        best_json = src / "best_checkpoint.json"
        if best_json.exists():
            shutil.copy(str(best_json), str(dst / "best_checkpoint.json"))

        print(f"Ready to resume from step {step}")
        return step
