from __future__ import annotations
"""src/preprocess.py – GSM8K loading & tokenisation."""

from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

CACHE_DIR = ".cache/"

# ---------------------------------------------------------------------------
# GSM8K formatting -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _format(ex: dict) -> Tuple[str, str]:
    q = ex["question"].strip()
    a = ex["answer"].strip()
    return f"Question: {q}\nAnswer:", a

# ---------------------------------------------------------------------------
# dataloaders ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_dataloaders(cfg, tok: PreTrainedTokenizerBase):
    """Return training & validation dataloaders according to *cfg*."""
    name = cfg.dataset.name
    ds_tr = load_dataset(name, split=cfg.dataset.splits.train, cache_dir=CACHE_DIR)
    ds_val = load_dataset(name, split=cfg.dataset.splits.validation, cache_dir=CACHE_DIR)
    max_len = cfg.dataset.text_max_length

    def _tok_single(ex):
        prompt, ans = _format(ex)
        prompt_ids = tok(prompt, truncation=True, max_length=max_len)
        full = tok(prompt + " " + ans, truncation=True, max_length=max_len)
        labels = full["input_ids"].copy()
        labels[: len(prompt_ids["input_ids"])] = [-100] * len(prompt_ids["input_ids"])
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    ds_tr = ds_tr.map(_tok_single, remove_columns=ds_tr.column_names, desc="tokenise-tr")
    ds_val = ds_val.map(_tok_single, remove_columns=ds_val.column_names, desc="tokenise-val")

    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100)
    tr_dl = DataLoader(
        ds_tr,
        batch_size=cfg.training.micro_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
    )
    val_dl = DataLoader(
        ds_val,
        batch_size=cfg.training.micro_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )
    return tr_dl, val_dl