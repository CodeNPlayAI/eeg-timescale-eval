# scripts/generate_gan_only.py
"""
generate_gan_only.py

2-channel (F7, F8) UNCONDITIONAL 1D WGAN-GP EEG generator.

PARITY CONTRACT (must match generate_diffusion_only.py exactly):
  - input           : real CSV columns [F7, F8] -> (2, N), fs=256
  - training window : train_win_s=10 -> win=2560 samples ; train_hop_s=5 -> hop=1280 (50% overlap)
  - normalization   : per-window per-channel z-score (zscore_per_window). Generator output is LINEAR
                      (no tanh). No de-normalization back to microvolts; output lives in z-scored space,
                      exactly like the diffusion model.
  - 180 s synthesis : generate Wgen = 1 + (N - win)//hop independent windows, then overlap-add
                      (Hanning, 50% COLA normalization) and trim to N. Output CSV = time, F7, F8.
  The three utilities window_view_np / zscore_per_window / overlap_add are COPIED VERBATIM from
  generate_diffusion_only.py so that windowing, normalization, and stitching are byte-for-byte identical.
  Only the generative mechanism (adversarial WGAN-GP) differs from the diffusion condition.

WHY WGAN-GP AND NOT THE ORIGINAL GAN:
  Algorithm origin  : Gulrajani et al., NeurIPS 2017 (WGAN-GP; Adam lr=1e-4, betas=(0,0.9), lambda=10).
  Peer-reviewed EEG : Panwar et al., IEEE TNSRE 2020 (WGAN-GP for raw single-/64-channel EEG time series).
  Recent peer EEG   : Havugimana et al., EGAN, Neural Computing and Applications 2025 (WGAN-GP derivative).
  (YARE-GAN, 2025, arXiv, resting-state WGAN-GP: auxiliary / design reference only.)
  We implement a single-head 2-channel unconditional WGAN-GP. EGAN's Topomap branch and ERP averaging
  are NOT used (that setup is 64ch/ERP and does not fit F7/F8 continuous resting-state generation).

DIAGNOSTICS (single subject, multiple seeds; run BEFORE any 30-subject pipeline integration):
  Because one subject yields only ~30 windows (50% overlap), this is a very small training set for a
  WGAN-GP. For each seed we record and report:
    - critic loss, generator loss, GP, and critic grad-norm @ interp (mean/std/min/max) history (+ curves)
    - generation diversity (pairwise L2 among generated windows: min/mean/median; per-dim std)
    - nearest-neighbour distance generated->real windows, raw L2 AND shape-normalized (memorization probe)
    - duplicate generation (fraction of pairs below dup_ratio x MEDIAN pairwise distance; mode collapse)
    - Pearson r_max memorization probe: for every generated window, the max correlation against ALL
      training windows. Reported as a distribution (mean/median/p75/p90/p95/p99/max), threshold
      fractions (r_max > 0.5 / 0.6 / 0.7), the nearest training-window index per generated window,
      and NN concentration (how many distinct training windows are hit, and the peak pile-up rate).
      Reference sets measured identically: training windows leave-one-out, and -- with --holdout --
      held-out real windows from a CONTIGUOUS time block separated by a guard interval.
    - persisted artifacts per seed: generator checkpoint (.pt), diagnostic windows + full correlation
      matrix (.npz), per-window r_max table (.csv), single-row health summary (.json and .csv)
    - NaN / Inf, flat (near-zero std) windows, extreme values
    - a light Welch-PSD sanity plot + rough 2-40 Hz log-log slope (NOT the authoritative metric;
      the authoritative PSD slope / DFA / ACF are produced later by evaluate.py / metrics.py)
  and an across-seed summary table.

Data interface only is matched to diffusion; WGAN-GP training hyper-parameters are independent of the
diffusion T/epochs and are meant to be tuned from the diagnostics (EGAN starting point: lr=1e-4,
critic:gen=5:1, lambda_gp=10, latent=100).

Usage (single subject, 3 seeds, diagnostics):
    python scripts/generate_gan_only.py \
        --input data/eeg_record_20251226_173831.csv \
        --out_dir runs/gan_diag \
        --fs 256 --duration_s 180 \
        --train_win_s 10 --train_hop_s 5 \
        --seeds "0 1 2" \
        --gen_steps 3000 --batch 16 --n_critic 5 --gp_lambda 10 \
        --latent 100 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# =====================================================================================
# VERBATIM from generate_diffusion_only.py  (do NOT modify -> guarantees interface parity)
# =====================================================================================
def window_view_np(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """x: (C, N) -> windows: (W, C, win)"""
    C, N = x.shape
    if N < win:
        raise ValueError(f"N={N} < win={win}")
    W = 1 + (N - win) // hop
    out = np.zeros((W, C, win), dtype=np.float32)
    for i in range(W):
        s = i * hop
        out[i] = x[:, s : s + win]
    return out


def zscore_per_window(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """w: (B, C, T) window-wise zscore per channel"""
    mu = w.mean(dim=-1, keepdim=True)
    sd = w.std(dim=-1, keepdim=True)
    return (w - mu) / (sd + eps)


def overlap_add(wins: np.ndarray, hop: int) -> np.ndarray:
    """
    wins: (W, C, T)
    return: (C, N)
    """
    W, C, T = wins.shape
    N = (W - 1) * hop + T
    out = np.zeros((C, N), dtype=np.float32)
    w = np.hanning(T).astype(np.float32)
    w = w / (w.sum() / hop + 1e-8)  # mild normalization

    for i in range(W):
        s = i * hop
        out[:, s : s + T] += wins[i] * w[None, :]
    return out


# =====================================================================================
# WGAN-GP model (2-channel, unconditional; LINEAR generator output; NO tanh / NO min-max)
# =====================================================================================
class Generator1D(nn.Module):
    """latent -> (2, win). Upsample(x2)+Conv1d blocks. Linear output (no activation)."""

    def __init__(self, win: int, latent: int = 100, base: int = 64,
                 init_len: int = 40, gen_norm: str = "batch"):
        super().__init__()
        assert win % init_len == 0, f"win={win} must be a multiple of init_len={init_len}"
        ratio = win // init_len
        n_ups = int(round(np.log2(ratio)))
        if init_len * (2 ** n_ups) != win:
            raise ValueError(
                f"win={win} is not init_len*2^k. Pick init_len so that win/init_len is a power of 2 "
                f"(e.g. win=2560 -> init_len=40, k=6)."
            )
        self.win = win
        self.latent = latent
        self.init_len = init_len

        c0 = base * 4  # 256
        self.fc = nn.Linear(latent, c0 * init_len)

        chans = [c0]
        for _ in range(n_ups):
            chans.append(max(base, chans[-1] // 2))
        # chans e.g. [256,128,64,64,64,64,64] for k=6

        blocks = []
        for i in range(n_ups):
            cin, cout = chans[i], chans[i + 1]
            blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            blocks.append(nn.Conv1d(cin, cout, kernel_size=5, padding=2))
            if gen_norm == "batch":
                blocks.append(nn.BatchNorm1d(cout))
            elif gen_norm == "group":
                blocks.append(nn.GroupNorm(min(8, cout), cout))
            # gen_norm == "none": no normalization
            blocks.append(nn.ReLU(inplace=True))
        self.up = nn.Sequential(*blocks)
        self.out_conv = nn.Conv1d(chans[-1], 2, kernel_size=5, padding=2)  # linear output

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, self.init_len)
        h = self.up(h)
        return self.out_conv(h)  # (B, 2, win), linear


class Critic1D(nn.Module):
    """(2, win) -> scalar. Strided Conv1d downsampling. NO BatchNorm (WGAN-GP GP requirement)."""

    def __init__(self, win: int, base: int = 64):
        super().__init__()
        layers = []
        cin = 2
        length = win
        chans = [base, base * 2, base * 2, base * 4, base * 4, base * 4]
        for cout in chans:
            layers.append(nn.Conv1d(cin, cout, kernel_size=5, stride=2, padding=2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            cin = cout
            length = (length + 2 * 2 - 5) // 2 + 1  # k=5,pad=2,stride=2
        self.body = nn.Sequential(*layers)
        self.flat_dim = cin * length
        self.head = nn.Linear(self.flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        h = h.reshape(h.size(0), -1)
        return self.head(h)  # (B, 1), no sigmoid


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                     device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (gp, grad_norm). grad_norm is the per-sample L2 norm of the critic gradient at the
    interpolated points. WGAN-GP softly pushes grad_norm toward 1; we RECORD the full distribution
    (mean/std/min/max) rather than treating gp -> 0 as a hard requirement."""
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=device)
    interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    score = critic(interp)
    grad = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad = grad.reshape(B, -1)
    grad_norm = grad.norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp, grad_norm


# =====================================================================================
# training
# =====================================================================================
def train_wgan_gp(data_z: torch.Tensor, win: int, device: str, args) -> tuple[nn.Module, dict]:
    """data_z: (W, 2, win) already per-window z-scored. Returns (G, history)."""
    W = data_z.size(0)
    G = Generator1D(win=win, latent=args.latent, base=args.base,
                    init_len=args.gen_init_len, gen_norm=args.gen_norm).to(device)
    C = Critic1D(win=win, base=args.base).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optC = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    hist = {"gen_step": [], "critic_loss": [], "gen_loss": [], "gp": [], "w_est": [],
            "gnorm_mean": [], "gnorm_std": [], "gnorm_min": [], "gnorm_max": []}
    # w_est = E[C(real)] - E[C(fake)] (Wasserstein estimate); gnorm_* = critic grad-norm stats @ interp

    G.train(); C.train()
    for gstep in range(1, args.gen_steps + 1):
        c_loss_v = gp_v = w_v = 0.0
        gn_mean = gn_std = gn_min = gn_max = 0.0
        for _ in range(args.n_critic):
            idx = torch.randint(0, W, (args.batch,), device=device)  # sampling WITH replacement (W is tiny)
            real = data_z[idx]
            z = torch.randn(args.batch, args.latent, device=device)
            with torch.no_grad():
                fake = G(z)
            c_real = C(real).mean()
            c_fake = C(fake).mean()
            gp, grad_norm = gradient_penalty(C, real, fake, device)
            c_loss = c_fake - c_real + args.gp_lambda * gp
            optC.zero_grad(set_to_none=True)
            c_loss.backward()
            optC.step()
            c_loss_v = float(c_loss.detach())
            gp_v = float(gp.detach())
            w_v = float((c_real - c_fake).detach())
            gn = grad_norm.detach()
            gn_mean = float(gn.mean()); gn_std = float(gn.std())
            gn_min = float(gn.min()); gn_max = float(gn.max())

        z = torch.randn(args.batch, args.latent, device=device)
        fake = G(z)
        g_loss = -C(fake).mean()
        optG.zero_grad(set_to_none=True)
        g_loss.backward()
        optG.step()

        if gstep % args.log_every == 0 or gstep == 1 or gstep == args.gen_steps:
            hist["gen_step"].append(gstep)
            hist["critic_loss"].append(c_loss_v)
            hist["gen_loss"].append(float(g_loss.detach()))
            hist["gp"].append(gp_v)
            hist["w_est"].append(w_v)
            hist["gnorm_mean"].append(gn_mean)
            hist["gnorm_std"].append(gn_std)
            hist["gnorm_min"].append(gn_min)
            hist["gnorm_max"].append(gn_max)
            if args.verbose:
                print(f"    [gan] step={gstep}/{args.gen_steps} "
                      f"c_loss={c_loss_v:.3f} g_loss={float(g_loss.detach()):.3f} "
                      f"gp={gp_v:.3f} W_est={w_v:.3f} "
                      f"gnorm(m/s/min/max)={gn_mean:.2f}/{gn_std:.2f}/{gn_min:.2f}/{gn_max:.2f}")
    return G, hist


@torch.no_grad()
def sample_windows(G: nn.Module, n: int, win: int, latent: int, device: str,
                   batch: int = 256) -> np.ndarray:
    G.eval()
    outs = []
    done = 0
    while done < n:
        b = min(batch, n - done)
        z = torch.randn(b, latent, device=device)
        outs.append(G(z).cpu().numpy().astype(np.float32))
        done += b
    return np.concatenate(outs, axis=0)  # (n, 2, win)


def generate_180s(G: nn.Module, win: int, hop: int, fs: float, duration_s: float,
                  latent: int, device: str) -> np.ndarray:
    """Same synthesis path as diffusion-only: independent windows -> overlap_add -> trim."""
    N = int(round(fs * duration_s))
    Wgen = 1 + (N - win) // hop
    if Wgen <= 0:
        raise ValueError("duration too short for windowing")
    gen_w = sample_windows(G, Wgen, win, latent, device)  # (Wgen, 2, win), z-scored space
    y = overlap_add(gen_w, hop=hop)  # (2, N_stitched)
    return y[:, :N]


# =====================================================================================
# diagnostics
# =====================================================================================
def _global_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: (C, N) -> per-channel global z-score (mirrors evaluate.py --zscore global)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + eps)


def _pairwise_dists(A: np.ndarray, sample_pairs: int = 20000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """A: (M, D). Returns an array of L2 distances over random off-diagonal pairs."""
    M = A.shape[0]
    if M < 2:
        return np.array([], dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    npairs = min(sample_pairs, M * (M - 1) // 2)
    i = rng.integers(0, M, size=npairs)
    j = rng.integers(0, M, size=npairs)
    mask = i != j
    i, j = i[mask], j[mask]
    return np.sqrt(((A[i] - A[j]) ** 2).sum(axis=1))


def _zscore_windows_np(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """w: (M, C, T) -> per-window per-channel z-score (shape-normalization; amplitude-invariant)."""
    mu = w.mean(axis=2, keepdims=True)
    sd = w.std(axis=2, keepdims=True)
    return (w - mu) / (sd + eps)


# ---------------------------------------------------------------------------------
# Pearson-correlation memorization probe
#
# For every QUERY window we take the maximum Pearson correlation against ALL training
# windows -> r_max. Because r_max is a max over the SAME set of reference training
# windows within a single run, r_max distributions from different query sets (generated
# windows, held-out real windows, training windows leave-one-out) can be compared on a
# common footing -- unlike a bare "minimum distance over all pairs", which depends on how
# many pairs were formed.
#
# This does NOT eliminate comparison-count effects. The query sets still differ in size
# (e.g. ~200 generated windows vs a handful of held-out windows), so the extreme-tail
# statistics (max, p99) remain sensitive to the number of query windows. Compare the
# central statistics (median, p75, p90) and treat any single extreme value as a trigger
# for further checking, not as evidence in itself.
#
# The correlation is computed on the flattened 2-channel window after per-channel
# z-scoring, so it is consistent with the existing shape-normalized NN columns:
#   r = 1 - d_shape^2 / (2 * C * T)
# Note the correspondence with the existing columns:
#   nn_gen2real_shape_min  <-> rmax_gen_max   (ONE extreme window, not a typical one)
#   nn_gen2real_shape_mean <-> rmax_gen_mean  (approximately; d vs d^2 Jensen gap)
# ---------------------------------------------------------------------------------
def _flat_unit(w: np.ndarray) -> np.ndarray:
    """(M,C,T) -> (M, C*T) rows centered to zero mean and unit L2 norm (exact Pearson via dot)."""
    a = _zscore_windows_np(w).reshape(w.shape[0], -1).astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / (n + 1e-12)


def pearson_rmax(query: np.ndarray, ref: np.ndarray,
                 exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """query:(M,C,T), ref:(R,C,T) -> (r_max (M,), nn_idx (M,), R_full (M,R)).
    exclude_self=True masks the diagonal (use when query IS ref, i.e. leave-one-out)."""
    Q = _flat_unit(query)
    Rf = _flat_unit(ref)
    R_full = Q @ Rf.T  # (M,R) Pearson correlations
    Rm = R_full.copy()
    if exclude_self:
        n = min(Rm.shape)
        Rm[np.arange(n), np.arange(n)] = -np.inf
    nn_idx = np.argmax(Rm, axis=1)
    r_max = Rm[np.arange(Rm.shape[0]), nn_idx]
    return r_max, nn_idx.astype(int), R_full


def _rmax_stats(r_max: np.ndarray, prefix: str) -> dict:
    """Distribution statistics of r_max plus threshold-exceedance fractions."""
    if r_max.size == 0:
        return {}
    out = {
        f"{prefix}_n": int(r_max.size),
        f"{prefix}_mean": float(np.mean(r_max)),
        f"{prefix}_median": float(np.median(r_max)),
        f"{prefix}_p75": float(np.percentile(r_max, 75)),
        f"{prefix}_p90": float(np.percentile(r_max, 90)),
        f"{prefix}_p95": float(np.percentile(r_max, 95)),
        f"{prefix}_p99": float(np.percentile(r_max, 99)),
        f"{prefix}_max": float(np.max(r_max)),
    }
    for thr in (0.5, 0.6, 0.7):
        out[f"{prefix}_frac_gt{thr:.1f}"] = float(np.mean(r_max > thr))
    return out


def _nn_concentration(nn_idx: np.ndarray, n_ref: int, prefix: str) -> dict:
    """How many distinct training windows are used as nearest neighbour, and how strongly
    the queries pile up onto a single training window (mode-collapse-toward-one-exemplar probe)."""
    if nn_idx.size == 0:
        return {}
    counts = np.bincount(nn_idx, minlength=n_ref)
    return {
        f"{prefix}_nn_unique": int((counts > 0).sum()),
        f"{prefix}_nn_unique_frac": float((counts > 0).sum() / max(1, n_ref)),
        f"{prefix}_nn_max_concentration": float(counts.max() / nn_idx.size),
        f"{prefix}_nn_argmax_train_idx": int(np.argmax(counts)),
    }


def _nn_distance(query: np.ndarray, ref: np.ndarray, block: int = 64) -> np.ndarray:
    """For each row in query (M,D), min L2 distance to any row in ref (R,D)."""
    M = query.shape[0]
    out = np.empty(M, dtype=np.float64)
    ref2 = (ref ** 2).sum(axis=1)  # (R,)
    for s in range(0, M, block):
        q = query[s:s + block]  # (b,D)
        # ||q-r||^2 = ||q||^2 - 2 q.r + ||r||^2
        d2 = (q ** 2).sum(axis=1, keepdims=True) - 2.0 * q @ ref.T + ref2[None, :]
        d2 = np.clip(d2, 0.0, None)
        out[s:s + block] = np.sqrt(d2.min(axis=1))
    return out


def run_diagnostics(G: nn.Module, data_z: torch.Tensor, real_full: np.ndarray,
                    hist: dict, win: int, hop: int, fs: float, duration_s: float,
                    seed: int, out_seed_dir: Path, args,
                    holdout_w: np.ndarray | None = None) -> dict:
    out_seed_dir.mkdir(parents=True, exist_ok=True)
    device = str(next(G.parameters()).device)
    rng = np.random.default_rng(seed)

    # --- (0) reference windows (real, z-scored) as flat vectors ---
    real_w = data_z.cpu().numpy()                      # (Wr, 2, win)
    Wr = real_w.shape[0]
    real_flat = real_w.reshape(Wr, -1)                 # (Wr, 2*win)

    # --- (1) generated windows for health checks ---
    gen_w = sample_windows(G, args.diag_n_gen, win, args.latent, device)  # (M,2,win)
    M = gen_w.shape[0]
    gen_flat = gen_w.reshape(M, -1)

    # --- NaN / Inf / flat / extreme ---
    nan_count = int(np.isnan(gen_w).sum())
    inf_count = int(np.isinf(gen_w).sum())
    perwin_std = gen_w.std(axis=2)                     # (M,2) std over time, per channel
    flat_windows = int((perwin_std < 1e-3).any(axis=1).sum())
    finite = gen_w[np.isfinite(gen_w)]
    extreme_frac = float((np.abs(finite) > args.extreme_thr).mean()) if finite.size else float("nan")
    amp_min = float(np.nanmin(finite)) if finite.size else float("nan")
    amp_max = float(np.nanmax(finite)) if finite.size else float("nan")
    amp_mean_winstd = float(np.nanmean(perwin_std))

    # --- diversity (generated) ---
    pd_gen = _pairwise_dists(gen_flat, rng=rng)
    gen_min_l2 = float(pd_gen.min()) if pd_gen.size else float("nan")
    gen_mean_l2 = float(pd_gen.mean()) if pd_gen.size else float("nan")
    gen_median_l2 = float(np.median(pd_gen)) if pd_gen.size else float("nan")
    gen_perdim_std_mean = float(gen_flat.std(axis=0).mean())

    # duplicate fraction: pairs closer than dup_ratio * MEDIAN pairwise distance (median, matching text)
    dup_frac = (float((pd_gen < args.dup_ratio * (gen_median_l2 + 1e-12)).mean())
                if pd_gen.size else float("nan"))

    # --- nearest neighbour: generated -> real ---
    def _self_excluded_nn_min(A: np.ndarray) -> float:
        n = A.shape[0]
        if n < 2:
            return float("nan")
        a2 = (A ** 2).sum(axis=1)
        best = np.inf
        for a in range(n):
            d2 = a2[a] - 2.0 * A @ A[a] + a2
            d2[a] = np.inf
            best = min(best, float(np.sqrt(max(0.0, d2.min()))))
        return best

    # (a) raw L2: generator output vs stored per-window z-scored real windows (amplitude-sensitive)
    nn_g2r_raw = _nn_distance(gen_flat, real_flat)
    nn_r2r_raw_min = _self_excluded_nn_min(real_flat)
    # (b) shape-normalized: channel-wise z-score BOTH generated and real windows, then NN (amplitude-invariant)
    gen_flat_shape = _zscore_windows_np(gen_w).reshape(M, -1)
    real_flat_shape = _zscore_windows_np(real_w).reshape(Wr, -1)
    nn_g2r_shape = _nn_distance(gen_flat_shape, real_flat_shape)
    nn_r2r_shape_min = _self_excluded_nn_min(real_flat_shape)

    # --- (1b) Pearson r_max memorization probe (generated / train-LOO / optional held-out) ---
    rmax_gen, nnidx_gen, R_gen = pearson_rmax(gen_w, real_w)
    rmax_tr, nnidx_tr, _ = pearson_rmax(real_w, real_w, exclude_self=True)
    mem = {}
    mem.update(_rmax_stats(rmax_gen, "rmax_gen"))
    mem.update(_nn_concentration(nnidx_gen, Wr, "rmax_gen"))
    mem.update(_rmax_stats(rmax_tr, "rmax_trainLOO"))
    rmax_hold = nnidx_hold = None
    if holdout_w is not None and len(holdout_w) > 0:
        rmax_hold, nnidx_hold, R_hold = pearson_rmax(holdout_w, real_w)
        mem.update(_rmax_stats(rmax_hold, "rmax_holdout"))
        mem.update(_nn_concentration(nnidx_hold, Wr, "rmax_holdout"))
        # the decisive contrast: generated vs held-out real, both measured against training windows
        mem["rmax_gen_minus_holdout_median"] = float(np.median(rmax_gen) - np.median(rmax_hold))

    # --- (1c) persist generator checkpoint + diagnostic windows for later re-analysis ---
    ckpt_path = out_seed_dir / f"generator_seed{seed}.pt"
    torch.save({
        "seed": seed,
        "state_dict": G.state_dict(),
        "arch": {"win": win, "latent": args.latent, "base": args.base,
                 "init_len": args.gen_init_len, "gen_norm": args.gen_norm},
        "data_interface": {"fs": fs, "win": win, "hop": hop,
                           "train_win_s": args.train_win_s, "train_hop_s": args.train_hop_s,
                           "duration_s": duration_s, "normalization": "per-window z-score",
                           "generator_output": "linear"},
        "train_cfg": {"gen_steps": args.gen_steps, "batch": args.batch, "n_critic": args.n_critic,
                      "gp_lambda": args.gp_lambda, "lr": args.lr,
                      "betas": (args.beta1, args.beta2)},
        "n_train_windows": int(Wr),
    }, ckpt_path)

    npz_path = out_seed_dir / f"diag_windows_seed{seed}.npz"
    npz_payload = {
        "gen_windows": gen_w.astype(np.float32),      # (M,2,win) generator output, z-scored space
        "train_windows": real_w.astype(np.float32),   # (Wr,2,win) per-window z-scored training windows
        "r_matrix_gen_vs_train": R_gen.astype(np.float32),  # (M,Wr) full Pearson matrix
        "rmax_gen": rmax_gen.astype(np.float32),
        "nn_idx_gen": nnidx_gen.astype(np.int32),
        "rmax_trainLOO": rmax_tr.astype(np.float32),
        "nn_idx_trainLOO": nnidx_tr.astype(np.int32),
        "fs": np.array([fs]), "win": np.array([win]), "hop": np.array([hop]),
    }
    if rmax_hold is not None:
        npz_payload["holdout_windows"] = holdout_w.astype(np.float32)
        npz_payload["r_matrix_holdout_vs_train"] = R_hold.astype(np.float32)
        npz_payload["rmax_holdout"] = rmax_hold.astype(np.float32)
        npz_payload["nn_idx_holdout"] = nnidx_hold.astype(np.int32)
    if not args.diag_no_save_windows:
        np.savez_compressed(npz_path, **npz_payload)

    # per-window table (human-readable, re-analyzable)
    pw = pd.DataFrame({"gen_idx": np.arange(M), "r_max": rmax_gen,
                       "nn_train_idx": nnidx_gen,
                       "nn_shape_dist": nn_g2r_shape, "nn_raw_dist": nn_g2r_raw})
    pw.to_csv(out_seed_dir / "rmax_per_window.csv", index=False)
    if rmax_hold is not None:
        pd.DataFrame({"holdout_idx": np.arange(len(rmax_hold)), "r_max": rmax_hold,
                      "nn_train_idx": nnidx_hold}).to_csv(
            out_seed_dir / "rmax_per_window_holdout.csv", index=False)

    # --- (2) light PSD sanity (NOT authoritative) ---
    y180 = generate_180s(G, win, hop, fs, duration_s, args.latent, device)  # (2,N)
    slope_gen = slope_real = float("nan")
    if _HAVE_SCIPY:
        gz_gen = _global_zscore(y180)
        gz_real = _global_zscore(real_full)
        nper = int(fs * 4)
        f_g, P_g = welch(gz_gen, fs=fs, nperseg=nper, axis=1)
        f_r, P_r = welch(gz_real, fs=fs, nperseg=nper, axis=1)
        Pg = P_g.mean(axis=0); Pr = P_r.mean(axis=0)

        def _slope(fq, P):
            m = (fq >= 2) & (fq <= 40) & (P > 0)
            if m.sum() < 5:
                return float("nan")
            b = np.polyfit(np.log10(fq[m]), np.log10(P[m]), 1)
            return float(b[0])

        slope_gen = _slope(f_g, Pg); slope_real = _slope(f_r, Pr)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        m = (f_g >= 1) & (f_g <= 45)
        ax.loglog(f_r[(f_r >= 1) & (f_r <= 45)], Pr[(f_r >= 1) & (f_r <= 45)], label="real (global-z)")
        ax.loglog(f_g[m], Pg[m], label="GAN (global-z)")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (a.u.)")
        ax.set_title(f"PSD sanity seed={seed}  slope2-40: real={slope_real:.2f} gan={slope_gen:.2f}")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_seed_dir / "psd_sanity.png", dpi=150); plt.close(fig)

    # --- (3) loss / gp / gradient-norm curves ---
    h = pd.DataFrame({k: hist[k] for k in
                      ["gen_step", "critic_loss", "gen_loss", "gp", "w_est",
                       "gnorm_mean", "gnorm_std", "gnorm_min", "gnorm_max"]})
    h.to_csv(out_seed_dir / "loss_history.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
    axes[0].plot(h["gen_step"], h["critic_loss"], label="critic loss")
    axes[0].plot(h["gen_step"], h["gen_loss"], label="generator loss")
    axes[0].plot(h["gen_step"], h["w_est"], label="W est = E[C(real)]-E[C(fake)]", linestyle="--")
    axes[0].set_xlabel("generator step"); axes[0].set_title(f"losses seed={seed}"); axes[0].legend()
    axes[1].plot(h["gen_step"], h["gp"], color="tab:red")
    axes[1].axhline(0.0, lw=0.8, color="k")
    axes[1].set_xlabel("generator step"); axes[1].set_ylabel("gradient penalty")
    axes[1].set_title("gradient penalty (diagnostic, not a hard target)")
    axes[2].plot(h["gen_step"], h["gnorm_mean"], color="tab:green", label="grad-norm mean")
    axes[2].fill_between(h["gen_step"], h["gnorm_min"], h["gnorm_max"],
                         color="tab:green", alpha=0.2, label="min-max")
    axes[2].axhline(1.0, lw=0.9, color="k", linestyle=":")
    axes[2].set_xlabel("generator step"); axes[2].set_ylabel("critic grad-norm @ interp")
    axes[2].set_title("gradient norm (WGAN-GP target ~1)"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out_seed_dir / "loss_gp_curves.png", dpi=150); plt.close(fig)

    # save an example generated window vs a real window
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    t = np.arange(win) / fs
    ax[0].plot(t, real_w[0, 0], lw=0.8); ax[0].set_title("example real window (F7, z-scored)")
    ax[1].plot(t, gen_w[0, 0], lw=0.8, color="tab:orange"); ax[1].set_title("example GAN window (F7, z-scored)")
    ax[1].set_xlabel("time (s)"); fig.tight_layout()
    fig.savefig(out_seed_dir / "example_windows.png", dpi=150); plt.close(fig)

    diag = {
        "seed": seed,
        "n_train_windows": int(Wr),
        "n_gen_windows_diag": int(M),
        "final_critic_loss": float(h["critic_loss"].iloc[-1]),
        "final_gen_loss": float(h["gen_loss"].iloc[-1]),
        "final_gp": float(h["gp"].iloc[-1]),
        "final_w_est": float(h["w_est"].iloc[-1]),
        "final_gnorm_mean": float(h["gnorm_mean"].iloc[-1]),
        "final_gnorm_std": float(h["gnorm_std"].iloc[-1]),
        "final_gnorm_min": float(h["gnorm_min"].iloc[-1]),
        "final_gnorm_max": float(h["gnorm_max"].iloc[-1]),
        "gen_div_mean_l2": gen_mean_l2,
        "gen_div_median_l2": gen_median_l2,
        "gen_div_min_l2": gen_min_l2,
        "gen_perdim_std_mean": gen_perdim_std_mean,
        "dup_frac_lt_%.2fxmedian" % args.dup_ratio: dup_frac,
        "nn_gen2real_raw_mean": float(nn_g2r_raw.mean()),
        "nn_gen2real_raw_min": float(nn_g2r_raw.min()),
        "nn_real2real_raw_min": nn_r2r_raw_min,
        "nn_gen2real_shape_mean": float(nn_g2r_shape.mean()),
        "nn_gen2real_shape_min": float(nn_g2r_shape.min()),
        "nn_real2real_shape_min": nn_r2r_shape_min,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "flat_windows": flat_windows,
        "extreme_frac_gt_%g" % args.extreme_thr: extreme_frac,
        "amp_min": amp_min,
        "amp_max": amp_max,
        "amp_mean_winstd": amp_mean_winstd,
        "psd_slope_2_40_gan": slope_gen,
        "psd_slope_2_40_real": slope_real,
    }
    diag.update(mem)  # r_max distribution stats, threshold fractions, NN concentration, hold-out
    with open(out_seed_dir / "health_summary.json", "w") as fp:
        json.dump(diag, fp, indent=2)
    pd.DataFrame([diag]).to_csv(out_seed_dir / "health_summary.csv", index=False)
    return diag


# =====================================================================================
# main
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Real CSV for training (columns F7, F8)")
    ap.add_argument("--out_dir", required=True, help="Directory for generated CSVs + diagnostics")
    ap.add_argument("--fs", type=float, default=256.0)
    ap.add_argument("--duration_s", type=float, default=180.0)

    # dataset windowing (MUST match diffusion-only)
    ap.add_argument("--train_win_s", type=float, default=10.0)
    ap.add_argument("--train_hop_s", type=float, default=5.0)

    # WGAN-GP
    ap.add_argument("--latent", type=int, default=100)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--gen_init_len", type=int, default=40, help="win must equal gen_init_len*2^k")
    ap.add_argument("--gen_norm", choices=["batch", "group", "none"], default="group")
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--gp_lambda", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.0)
    ap.add_argument("--beta2", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gen_steps", type=int, default=3000, help="number of generator updates")
    ap.add_argument("--log_every", type=int, default=100)

    # diagnostics
    ap.add_argument("--seeds", type=str, default="0 1 2")
    ap.add_argument("--diag_n_gen", type=int, default=200)
    ap.add_argument("--dup_ratio", type=float, default=0.1, help="dup if pair dist < ratio*mean pairdist")
    ap.add_argument("--extreme_thr", type=float, default=8.0, help="|z| threshold for extreme values")
    ap.add_argument("--diag_no_save_windows", action="store_true",
                    help="skip writing diag_windows_seed*.npz (use for the large production run; "
                         "the generator checkpoint is still saved so windows can be regenerated)")

    # hold-out memorization control: CONTIGUOUS time blocks, never a random window split.
    # Training uses [0, split); the held-out block starts at split+guard so that no sample is
    # shared with any training window (guard >= one window length is recommended).
    ap.add_argument("--holdout", action="store_true",
                    help="enable contiguous-block hold-out memorization diagnostic")
    ap.add_argument("--holdout_frac", type=float, default=0.25,
                    help="fraction of the recording reserved (from the end) for the hold-out block")
    ap.add_argument("--guard_s", type=float, default=10.0,
                    help="guard interval (s) between the training block and the hold-out block")
    ap.add_argument("--verbose", action="store_true")

    # generation output
    ap.add_argument("--save_180s", action="store_true", help="also write the 180 s generated CSV per seed")
    args = ap.parse_args()

    fs = float(args.fs)
    win = int(round(fs * args.train_win_s))
    hop = int(round(fs * args.train_hop_s))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    for ch in ("F7", "F8"):
        if ch not in df.columns:
            raise ValueError(f"Missing column: {ch}")
    x = df[["F7", "F8"]].values.T.astype(np.float32)  # (2, N)
    real_full = x.copy()

    # ---- windowing (optionally with a CONTIGUOUS hold-out block separated by a guard) ----
    holdout_wins = None
    if args.holdout:
        N_all = x.shape[1]
        guard = int(round(fs * args.guard_s))
        split = int(round(N_all * (1.0 - args.holdout_frac)))
        # snap `split` down so the last training window ends exactly at or before `split`
        n_tr_w = 1 + max(0, (split - win)) // hop
        split = (n_tr_w - 1) * hop + win
        hold_start = split + guard
        if hold_start + win > N_all:
            raise ValueError(
                f"hold-out block too short: need >= {win} samples after split+guard "
                f"({hold_start}), but only {N_all - hold_start} available. "
                f"Lower --guard_s or raise --holdout_frac.")
        wins = window_view_np(x[:, :split], win=win, hop=hop)
        holdout_wins = window_view_np(x[:, hold_start:], win=win, hop=hop)
        holdout_wins = _zscore_windows_np(holdout_wins)  # same normalization as training windows
        print(f"[holdout] CONTIGUOUS split: train=[0,{split}) ({split/fs:.1f}s, {wins.shape[0]} win) | "
              f"guard={guard} ({args.guard_s:.1f}s) | holdout=[{hold_start},{N_all}) "
              f"({(N_all-hold_start)/fs:.1f}s, {holdout_wins.shape[0]} win)")
        print(f"[holdout] no sample is shared between the two blocks "
              f"(guard {'>=' if guard >= win else '<'} one window length {win})")
    else:
        wins = window_view_np(x, win=win, hop=hop)         # (W, 2, win)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        else "cpu")

    print(f"[cfg] fs={fs} win={win}({args.train_win_s}s) hop={hop}({args.train_hop_s}s) "
          f"train_windows={wins.shape[0]} device={device}")
    print(f"[cfg] WGAN-GP latent={args.latent} n_critic={args.n_critic} lambda={args.gp_lambda} "
          f"lr={args.lr} batch={args.batch} gen_steps={args.gen_steps} gen_norm={args.gen_norm}")

    seeds = [int(s) for s in str(args.seeds).split()]
    rows = []
    for seed in seeds:
        print(f"\n===== SEED {seed} =====")
        np.random.seed(seed); torch.manual_seed(seed)

        data_z = zscore_per_window(torch.tensor(wins, device=device))  # (W,2,win) z-scored
        G, hist = train_wgan_gp(data_z, win=win, device=device, args=args)

        seed_dir = out_root / f"seed{seed}"
        diag = run_diagnostics(G, data_z, real_full, hist, win, hop, fs,
                               args.duration_s, seed, seed_dir, args,
                               holdout_w=holdout_wins)
        rows.append(diag)

        if args.save_180s:
            y = generate_180s(G, win, hop, fs, args.duration_s, args.latent, device)
            N = y.shape[1]
            t = np.arange(N, dtype=np.float32) / fs
            out_csv = out_root / f"gan_3min_gan_seed{seed}.csv"
            pd.DataFrame({"time": t, "F7": y[0], "F8": y[1]}).to_csv(out_csv, index=False)
            print(f"[OK] 180s CSV: {out_csv}")

    summ = pd.DataFrame(rows)
    summ_path = out_root / "diag_seed_summary.csv"
    summ.to_csv(summ_path, index=False)
    try:
        (out_root / "diag_seed_summary.md").write_text(summ.to_markdown(index=False), encoding="utf-8")
    except Exception:
        pass
    print("\n[OK] per-seed diagnostics + curves under:", out_root)
    print("[OK] across-seed summary:", summ_path)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\n", summ.to_string(index=False))

    # quick interpretation hints
    print("\n[how to read]")
    print("  - gradient norm @ interp (green) should concentrate near 1 with bounded spread; the GP")
    print("    magnitude itself is NOT required to reach 0 (it is a soft 1-Lipschitz regularizer).")
    print("  - W_est (E[C(real)]-E[C(fake)]) should stay finite and not diverge; a collapsing W_est often")
    print("    tracks mode collapse.")
    print("  - gen_div_min_l2 ~ 0 or high dup_frac (median-based) => mode collapse (near-duplicate outputs).")
    print("  - nn_gen2real_shape_min << nn_real2real_shape_min => shape-level memorization (amplitude-")
    print("    invariant); the raw-L2 version is amplitude-sensitive and complements it.")
    print("  - flat_windows>0, nan/inf>0, or extreme_frac high => degenerate generation.")
    print("  - r_max probe: compare rmax_gen_* against rmax_trainLOO_* and (with --holdout)")
    print("    rmax_holdout_*. All are maxima over the SAME training windows, so they share a")
    print("    common footing, but the query sets differ in size, so max/p99 remain sensitive to")
    print("    the number of query windows -- judge on median/p75/p90, never on a single extreme.")
    print("    Memorization is indicated when the generated r_max distribution sits clearly ABOVE")
    print("    the held-out real distribution measured in the same run; if the two agree, a high")
    print("    r_max is a property of this data regime rather than evidence of copying.")
    print("    rmax_gen_nn_max_concentration near 1.0 means the generator piles onto one exemplar.")
    print("  - psd_slope_2_40_gan vs real is a SANITY check only; authoritative slope/DFA/ACF come")
    print("    from evaluate.py / metrics.py after pipeline integration.")


if __name__ == "__main__":
    main()
