"""src/main.py – orchestration wrapper launching *src.train* via Hydra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="config")
def main(cfg):  # noqa: ANN001  – Hydra injects DictConfig
    runs_dir = Path(__file__).resolve().parent.parent / "config" / "runs"
    run_cfg_path = runs_dir / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Unknown run id: {cfg.run}")

    overrides = []
    if cfg.mode == "trial":
        overrides += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
            "training.log_every_steps=1",
        ]
    elif cfg.mode == "full":
        overrides.append("wandb.mode=online")
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    overrides.append(f"results_dir={cfg.results_dir}")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"mode={cfg.mode}",
        *overrides,
    ]
    print("[main] launch →", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()