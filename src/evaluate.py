"""src/evaluate.py – independent evaluation & visualisation.
Fetches runs from WandB and produces per-run as well as aggregated reports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")  # enable head-less backend on CI

PRIMARY_METRIC = "GSM8K dev exact-match accuracy"

# ---------------------------------------------------------------------------
# utilities -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(path)

# ---------------------------------------------------------------------------
# per-run processing ---------------------------------------------------------
# ---------------------------------------------------------------------------

def _process_run(run_id: str, api, entity: str, project: str, out_dir: Path) -> Dict:
    run = api.run(f"{entity}/{project}/{run_id}")
    hist: pd.DataFrame = run.history(keys=None, pandas=True)
    summary = dict(run.summary)
    cfg = dict(run.config)

    _ensure(out_dir)

    # ----------- save metrics.json ----------------------------------------
    metrics_payload = {
        "config": cfg,
        "summary": summary,
        "history": hist.to_dict(orient="list") if not hist.empty else {},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    print(out_dir / "metrics.json")

    # ----------- learning curve ------------------------------------------
    if not hist.empty and {"train_loss", "val_acc"}.issubset(hist.columns):
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=hist, x="_step", y="train_loss", label="train_loss", ax=ax)
        sns.lineplot(data=hist, x="_step", y="val_acc", label="val_acc", ax=ax)
        ax.set_xlabel("step")
        ax.set_ylabel("loss / acc")
        ax.set_title(f"Learning curves – {run_id}")
        _save_fig(fig, out_dir / f"{run_id}_learning_curve.pdf")

    # ----------- confusion matrix (binary: correct vs wrong) --------------
    tot = int(summary.get("val_total_examples", 0))
    cor = int(summary.get("val_correct_examples", 0))
    if tot > 0:
        y_true = np.ones(tot, dtype=int)
        y_pred = np.array([1] * cor + [0] * (tot - cor))
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(["correct", "wrong"], rotation=0)
        ax.set_yticklabels(["correct", "wrong"], rotation=0)
        ax.set_title(f"Confusion – {run_id}")
        _save_fig(fig, out_dir / f"{run_id}_confusion_matrix.pdf")

    return {"run_id": run_id, "summary": summary}

# ---------------------------------------------------------------------------
# aggregation ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _aggregate(collected: List[Dict], comp_dir: Path):
    _ensure(comp_dir)
    metrics_by_key: Dict[str, Dict[str, float]] = {}
    for rec in collected:
        rid = rec["run_id"]
        for k, v in rec["summary"].items():
            if isinstance(v, (int, float)):
                metrics_by_key.setdefault(k, {})[rid] = float(v)

    # ---- find best proposed & baseline -----------------------------------
    proposed_runs = {
        k: v for k, v in metrics_by_key.get(PRIMARY_METRIC, {}).items() if "proposed" in k or "KDLS" in k
    }
    baseline_runs = {
        k: v for k, v in metrics_by_key.get(PRIMARY_METRIC, {}).items() if "baseline" in k or "comparative" in k
    }
    best_prop_id = max(proposed_runs, key=proposed_runs.get) if proposed_runs else None
    best_base_id = max(baseline_runs, key=baseline_runs.get) if baseline_runs else None

    gap = None
    if best_prop_id and best_base_id:
        gap = (
            (proposed_runs[best_prop_id] - baseline_runs[best_base_id])
            / baseline_runs[best_base_id]
            * 100.0
        )

    # ---- write aggregated_metrics.json -----------------------------------
    out = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": metrics_by_key,
        "best_proposed": {
            "run_id": best_prop_id,
            "value": proposed_runs.get(best_prop_id) if best_prop_id else None,
        },
        "best_baseline": {
            "run_id": best_base_id,
            "value": baseline_runs.get(best_base_id) if best_base_id else None,
        },
        "gap": gap,
    }
    (comp_dir / "aggregated_metrics.json").write_text(json.dumps(out, indent=2))
    print(comp_dir / "aggregated_metrics.json")

    # ---- primary metric bar-chart ----------------------------------------
    if PRIMARY_METRIC in metrics_by_key:
        labels, values = zip(*metrics_by_key[PRIMARY_METRIC].items())
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 4))
        sns.barplot(x=list(labels), y=list(values), ax=ax, palette="deep")
        ax.set_ylabel(PRIMARY_METRIC)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        for i, v in enumerate(values):
            ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom")
        ax.set_title("Cross-run comparison")
        _save_fig(fig, comp_dir / "comparison_primary_metric.pdf")

    # ---- statistical test -----------------------------------------------
    if best_prop_id and best_base_id:
        t, p = stats.ttest_ind(
            np.array([proposed_runs[best_prop_id]]),
            np.array([baseline_runs[best_base_id]]),
            equal_var=False,
        )
        (comp_dir / "ttest.txt").write_text(f"t={t:.4f}, p={p:.3e}\n")
        print(comp_dir / "ttest.txt")

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("run_ids", help="JSON list, e.g. '[\"run1\",\"run2\"]'")
    args = ap.parse_args()

    root = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    # --------------- global WandB credentials -----------------------------
    import yaml

    glob_cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "config.yaml").read_text()
    )
    entity = glob_cfg["wandb"]["entity"]
    project = glob_cfg["wandb"]["project"]

    api = wandb.Api()
    collected = []
    for rid in run_ids:
        collected.append(_process_run(rid, api, entity, project, root / rid))
    _aggregate(collected, root / "comparison")


if __name__ == "__main__":
    main()