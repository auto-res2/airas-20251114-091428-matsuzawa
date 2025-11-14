from __future__ import annotations
"""src/train.py – single-run executor for KDLS-QLoRA experiments.
This file launches *one* training run, handles Optuna (optional), and logs all
metrics to Weights & Biases.  It is fully executable and integrates with Hydra.
"""

import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, MutableMapping

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Add repository root to sys.path so that `src.*` imports work even after
# Hydra changes the working directory.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# pylint: disable=wrong-import-position
import wandb  # noqa: E402 – after sys.path patch
from src.model import (  # noqa: E402 – local import
    KDLSAdamW,
    build_model_and_tokenizer,
    mark_only_lora_as_trainable,
)
from src.preprocess import get_dataloaders  # noqa: E402
# pylint: enable=wrong-import-position

CACHE_DIR = ".cache/"

# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _to_ns(obj):  # ΩConf → SimpleNamespace (recursive)
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=False)
    if isinstance(obj, MutableMapping):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def _set_by_dotted(cfg, dotted_key: str, value):
    """Assign *value* into *cfg* at key expressed in dotted notation.

    Works for both `omegaconf.DictConfig` and plain dict / SimpleNamespace.
    """
    # Handle DictConfig ------------------------------------------------------
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, dotted_key, value, merge=True)
        OmegaConf.set_struct(cfg, True)
        return

    # Handle dict / SimpleNamespace -----------------------------------------
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if isinstance(cur, SimpleNamespace):
            if not hasattr(cur, k):
                setattr(cur, k, {})
            cur = getattr(cur, k)
        else:  # plain dict
            cur = cur.setdefault(k, {})
    if isinstance(cur, SimpleNamespace):
        setattr(cur, keys[-1], value)
    else:
        cur[keys[-1]] = value


def _suggest_from_space(trial: optuna.Trial, dotted_key: str, space: Dict):
    """Sample one hyper-parameter from an Optuna search-space description."""
    stype = space["type"].lower()
    if stype == "uniform":
        return trial.suggest_float(dotted_key, float(space["low"]), float(space["high"]))
    if stype == "loguniform":
        return trial.suggest_float(
            dotted_key, float(space["low"]), float(space["high"]), log=True
        )
    if stype == "categorical":
        return trial.suggest_categorical(dotted_key, list(space["choices"]))
    raise ValueError(f"Unknown search-space type: {stype}")

# ---------------------------------------------------------------------------
# GSM8K evaluation helpers ---------------------------------------------------
# ---------------------------------------------------------------------------

def _extract_num(text: str) -> str:
    """Extract first number appearing in *text* (GSM8K answer format)."""
    import re

    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", text)
    return m.group(0).replace(",", "") if m else ""


def _evaluate(model, val_dl, tok, device, *, max_batches: int | None = None):
    """Run validation: compute loss + exact-match accuracy on GSM8K."""
    model.eval()
    tot_loss = 0.0
    n_batches = 0
    seen = 0
    correct = 0
    predictions = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            out = model(**batch)
            tot_loss += out.loss.item()
            n_batches += 1

            gen = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=32,
                do_sample=False,
            )
            prompt_lens = (labels == -100).sum(dim=1)
            bs = labels.size(0)
            for i in range(bs):
                gold = _extract_num(
                    tok.decode(labels[i][labels[i] != -100], skip_special_tokens=True)
                )
                pred_txt = tok.decode(gen[i][prompt_lens[i] :], skip_special_tokens=True)
                pred = _extract_num(pred_txt)
                ok = gold != "" and pred == gold
                correct += int(ok)
                seen += 1
                if i == 0:  # keep dataset small in WandB artifact
                    predictions.append({"gold": gold, "pred": pred, "correct": ok})
            if max_batches and (b_idx + 1) >= max_batches:
                break

    model.train()
    avg_loss = tot_loss / max(1, n_batches)
    acc = correct / max(1, seen)
    return avg_loss, acc, predictions, seen, correct

# ---------------------------------------------------------------------------
# single training fit --------------------------------------------------------
# ---------------------------------------------------------------------------

def _single_fit(cfg):
    """One full training run (with or without KDLS) – returns best dev accuracy."""

    # ---------------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------------
    torch.manual_seed(int(cfg.training.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Build model, tokenizer, dataloaders
    # ---------------------------------------------------------------------
    model, tok, lora_modules = build_model_and_tokenizer(cfg, device)
    mark_only_lora_as_trainable(model)
    train_dl, val_dl = get_dataloaders(cfg, tok)

    # ---------------------------------------------------------------------
    # Optimiser (AdamW or KDLS-AdamW) and scheduler (optional cosine)
    # ---------------------------------------------------------------------
    optim_kwargs = dict(
        lr=float(cfg.training.optimizer.get("learning_rate", 1.0)),
        betas=tuple(float(x) for x in cfg.training.optimizer.betas),
        weight_decay=float(cfg.training.optimizer.get("weight_decay", 0.0)),
        eps=1e-8,
    )
    use_kdls = bool(getattr(cfg.model, "kdls", {}).get("enabled", False))

    if use_kdls:
        optimizer = KDLSAdamW(
            [p for p in model.parameters() if p.requires_grad],
            lora_modules=lora_modules,
            kdls_cfg=_to_ns(cfg.model.kdls),
            **optim_kwargs,
        )
        scheduler = None  # step-size comes from KDLS line search
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], **optim_kwargs
        )
        scheduler = None
        if cfg.training.get("lr_scheduler") and cfg.training.lr_scheduler.get("type") == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                optimizer, T_max=len(train_dl) * cfg.training.epochs
            )

    # ---------------------------------------------------------------------
    # WandB initialisation -------------------------------------------------
    # ---------------------------------------------------------------------
    wandb_run = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=str(cfg.run_id),
            mode=cfg.wandb.mode,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"[wandb] {wandb_run.url}")

    # ---------------------------------------------------------------------
    # Training loop --------------------------------------------------------
    # ---------------------------------------------------------------------
    acc_steps = cfg.training.gradient_accumulation_steps
    global_step = 0
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(cfg.training.epochs):
        running_loss = 0.0
        for b_idx, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / acc_steps
            loss.backward()
            running_loss += loss.item()

            if (b_idx + 1) % acc_steps == 0:
                if cfg.training.clipping.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.clipping.max_grad_norm
                    )
                if use_kdls:
                    optimizer.step(loss=loss.detach() * acc_steps)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                global_step += 1

                # frequent logging --------------------------------------
                if global_step % cfg.training.log_every_steps == 0 and wandb_run:
                    wandb.log(
                        {
                            "train_loss": running_loss / cfg.training.log_every_steps,
                            "lr": optimizer.param_groups[0]["lr"],
                            "global_step": global_step,
                        },
                        step=global_step,
                    )
                    running_loss = 0.0

                # truncate compute in trial mode ------------------------
                if cfg.mode == "trial" and global_step >= 2:
                    break

        # ------------------- validation each epoch -----------------------
        val_loss, val_acc, preds, total, correct = _evaluate(
            model,
            val_dl,
            tok,
            device,
            max_batches=(2 if cfg.mode == "trial" else None),
        )
        best_val_acc = max(best_val_acc, val_acc)

        if wandb_run:
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                },
                step=global_step,
            )
            preds_path = Path(wandb.run.dir) / f"val_preds_ep{epoch+1}.json"
            preds_path.write_text(json.dumps(preds, indent=2))
            wandb.save(str(preds_path))

        if cfg.mode == "trial":
            break  # only 1 epoch in trial mode

    runtime = time.time() - start_time

    # --------------------- WandB summary -----------------------------------
    if wandb_run:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["GSM8K dev exact-match accuracy"] = best_val_acc
        wandb.summary["val_total_examples"] = int(total)
        wandb.summary["val_correct_examples"] = int(correct)
        wandb.summary["total_train_time_sec"] = runtime
        wandb.finish()

    return best_val_acc, runtime

# ---------------------------------------------------------------------------
# Optuna wrapper -------------------------------------------------------------
# ---------------------------------------------------------------------------

def _fit_with_optuna(cfg):
    """Hyper-parameter search with Optuna (no WandB per-trial)."""
    base = copy.deepcopy(cfg)

    def objective(trial: optuna.Trial):
        cfg_trial = copy.deepcopy(base)
        for dotted_key, space in cfg_trial.optuna.search_space.items():
            _set_by_dotted(
                cfg_trial,
                dotted_key,
                _suggest_from_space(trial, dotted_key, space),
            )
        cfg_trial.wandb.mode = "disabled"
        cfg_trial.optuna.n_trials = 0
        cfg_trial.run_id = f"{cfg.run_id}-trial{trial.number}"
        acc, _ = _single_fit(cfg_trial)
        return acc

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=False)

    # -------- train once with the best params & WandB logging --------------
    best_cfg = copy.deepcopy(base)
    for k, v in study.best_trial.params.items():
        _set_by_dotted(best_cfg, k, v)
    best_cfg.optuna.n_trials = 0
    best_cfg.run_id = str(cfg.run_id)
    _single_fit(best_cfg)

# ---------------------------------------------------------------------------
# public entry-point ---------------------------------------------------------
# ---------------------------------------------------------------------------

def train(cfg):
    """Top-level trainer used by *src.main* as well as direct CLI."""
    # mode-specific tweaks --------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # choose optimisation route -------------------------------------------
    if cfg.optuna.n_trials and cfg.mode == "full":
        _fit_with_optuna(cfg)
    else:
        _single_fit(cfg)

# ---------------------------------------------------------------------------
# Hydra CLI entry ------------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def _hydra_entry(cfg):  # pragma: no cover – CLI entry point
    runs_dir = _PROJECT_ROOT / "config" / "runs"
    run_cfg_path = runs_dir / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Unknown run-id: {cfg.run}")

    run_cfg = OmegaConf.load(run_cfg_path)
    merged = OmegaConf.merge(cfg, run_cfg)
    merged.run_id = run_cfg.get("run_id", cfg.run)
    train(merged)


if __name__ == "__main__":  # pragma: no cover – python -m src.train
    _hydra_entry()