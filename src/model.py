"""src/model.py – model construction & KDLS-AdamW optimiser (bug-fixed)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

CACHE_DIR = ".cache/"

# ---------------------------------------------------------------------------
# KDLS statistics buffers ----------------------------------------------------
# ---------------------------------------------------------------------------

def _attach_kdls_buffers(module: nn.Module, rank: int, beta: float):
    """Register running covariance buffers X̄ & Ḡ and hooks on *module*."""
    module.register_buffer("kdls_X", torch.zeros(rank, rank), persistent=False)
    module.register_buffer("kdls_G", torch.zeros(rank, rank), persistent=False)
    module.kdls_beta = beta

    # ---------- forward hook: update input covariance ---------------------
    def _fwd(mod, inp, _out):  # noqa: ANN001
        x = inp[0].detach().float().view(-1, inp[0].shape[-1])  # (B,d_in)
        with torch.no_grad():
            proj = x @ mod.lora_A.weight.T  # (B,r)   A: (r,d_in)
            cov = (proj.T @ proj) / proj.size(0)  # (r,r)
            mod.kdls_X.mul_(mod.kdls_beta).add_(cov, alpha=1 - mod.kdls_beta)

    # ---------- backward hook: update grad covariance ---------------------
    def _bwd(mod, _gin, gout):  # noqa: ANN001
        g = gout[0].detach().float().view(-1, gout[0].shape[-1])  # (B,d_out)
        with torch.no_grad():
            proj = g @ mod.lora_B.weight  # (B,r)   B: (d_out,r)
            cov = (proj.T @ proj) / proj.size(0)  # (r,r)
            mod.kdls_G.mul_(mod.kdls_beta).add_(cov, alpha=1 - mod.kdls_beta)

    module.register_forward_hook(_fwd)
    module.register_full_backward_hook(_bwd)

# ---------------------------------------------------------------------------
# builder --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # quantisation via bitsandbytes ---------------------------------------
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4" if cfg.model.quantization.scheme == "nf4" else "fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )

    lcfg = LoraConfig(
        r=cfg.model.adapter.rank,
        lora_alpha=cfg.model.adapter.alpha,
        lora_dropout=cfg.model.adapter.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)

    # KDLS buffers ---------------------------------------------------------
    if getattr(cfg.model, "kdls", {}).get("enabled", False):
        for m in [m for m in model.modules() if hasattr(m, "lora_A")]:
            _attach_kdls_buffers(m, lcfg.r, cfg.model.kdls.beta)

    model.to(device)
    return model, tok, [m for m in model.modules() if hasattr(m, "lora_A")]

# ---------------------------------------------------------------------------
# freeze everything except LoRA params ---------------------------------------
# ---------------------------------------------------------------------------

def mark_only_lora_as_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad = any(t in n for t in ("lora_A", "lora_B"))

# ---------------------------------------------------------------------------
# KDLS-AdamW optimiser -------------------------------------------------------
# ---------------------------------------------------------------------------

class KDLSAdamW(torch.optim.Optimizer):
    """AdamW pre-conditioner + Kronecker-Diagonal Line Search."""

    def __init__(
        self,
        params,
        *,
        lora_modules: List[nn.Module],
        kdls_cfg: SimpleNamespace,
        lr: float = 1.0,
        betas=(0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        self.lora_mods = lora_modules
        self.cfg = kdls_cfg
        self.alpha_ema = 1.0

    # ---------------------------------------------------------------------
    def _adam_pass(self):
        """Compute standard AdamW *pre-conditioned* update and store in .grad_precond."""
        s1 = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p.data, dtype=torch.float32)
                m, v = state["m"], state["v"]
                state["step"] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                upd = lr * m_hat / (v_hat.sqrt().add_(eps))
                if wd:
                    upd = upd + wd * p.data
                p.grad_precond = upd  # stash for KDLS
                s1 += float((grad * upd).sum().item())
        return s1

    # ---------------------------------------------------------------------
    def _curvature_term(self):
        """Compute q = tr(BΔᵀ Ḡ BΔ) · tr(AΔᵀ X̄ AΔ)."""
        q = 0.0
        for mod in self.lora_mods:
            Aupd = getattr(mod.lora_A.weight, "grad_precond", None)
            Bupd = getattr(mod.lora_B.weight, "grad_precond", None)
            if Aupd is None or Bupd is None:
                continue
            XA = mod.kdls_X  # (r,r) – float32
            GA = mod.kdls_G  # (r,r)

            # cast updates to fp32 for numerical stability
            Aupd = Aupd.float()
            Bupd = Bupd.float()

            # traces -----------------------------------------------------
            trace_A = float(((Aupd @ Aupd.T) * XA).sum().item())  # r×r element-wise
            trace_B = float(((Bupd.T @ Bupd) * GA).sum().item())  # r×r
            q += trace_A * trace_B
        return q

    # ---------------------------------------------------------------------
    def step(self, *, loss: torch.Tensor, closure=None):  # noqa: D401, ANN001
        """Perform one optimisation step; *loss* is current value ℓ₀."""
        if loss is None:
            raise RuntimeError("KDLSAdamW.step() requires current loss value")

        s1 = self._adam_pass()
        q = self._curvature_term()

        disc = s1 * s1 - 2.0 * q * float(loss.item())
        disc = max(disc, 0.0)  # safeguard
        denom = q + getattr(self.cfg, "epsilon", 1e-12)
        alpha_star = (-s1 + math.sqrt(disc)) / denom if denom > 0.0 else 1.0
        alpha_star = max(0.0, min(self.cfg.polyak_max_scale, alpha_star))
        self.alpha_ema = (
            self.alpha_ema * self.cfg.polyak_ema + alpha_star * (1 - self.cfg.polyak_ema)
        )
        alpha = self.alpha_ema

        # apply update ----------------------------------------------------
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad_precond, alpha=-alpha)
        return alpha