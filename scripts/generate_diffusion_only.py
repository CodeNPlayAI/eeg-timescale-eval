# scripts/generate_diffusion_only.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# utils
# ----------------------------
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

def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """
    t: (B,) int64
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, device=device).float() / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ----------------------------
# model (tiny but "t is used")
# ----------------------------
class ResBlock1D(nn.Module):
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.tproj = nn.Linear(tdim, ch)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.gn1(x)))
        # inject time embedding as bias
        h = h + self.tproj(temb).unsqueeze(-1)
        h = self.conv2(F.silu(self.gn2(h)))
        return x + h


class TinyUNet1D(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 64, tdim: int = 64):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock1D(base, tdim)
        self.rb2 = ResBlock1D(base, tdim)
        self.out_conv = nn.Conv1d(base, in_ch, 3, padding=1)
        self.tdim = tdim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = sinusoidal_time_embedding(t, dim=self.tdim)
        h = self.in_conv(x)
        h = self.rb1(h, temb)
        h = self.rb2(h, temb)
        return self.out_conv(h)


# ----------------------------
# diffusion (epsilon-pred DDPM)
# ----------------------------
def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)

class DDPM:
    def __init__(self, T: int, device: str):
        self.T = T
        self.device = device
        self.betas = linear_beta_schedule(T).to(device)
        self.alphas = (1.0 - self.betas)
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # sqrt(a_hat)*x0 + sqrt(1-a_hat)*eps
        a_hat = self.alpha_hat[t].view(-1, 1, 1)
        return torch.sqrt(a_hat) * x0 + torch.sqrt(1.0 - a_hat) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: tuple[int, int, int]) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for ti in reversed(range(self.T)):
            t = torch.full((shape[0],), ti, device=self.device, dtype=torch.long)
            eps = model(x, t)
            alpha = self.alphas[ti]
            a_hat = self.alpha_hat[ti]
            beta = self.betas[ti]
            if ti > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - a_hat) * eps) + torch.sqrt(beta) * z
        return x


# ----------------------------
# overlap-add stitching
# ----------------------------
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


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Real CSV for training")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--fs", type=float, default=256.0)
    ap.add_argument("--duration_s", type=float, default=180.0)

    # dataset windowing
    ap.add_argument("--train_win_s", type=float, default=10.0)
    ap.add_argument("--train_hop_s", type=float, default=5.0)

    # diffusion/training
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=800)   # small dataset -> overfit OK for PoC
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="", help="Tag string appended to output filename to avoid overwriting")

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.input)
    x = df[["F7", "F8"]].values.T.astype(np.float32)  # (2, N)

    fs = float(args.fs)
    win = int(round(fs * args.train_win_s))
    hop = int(round(fs * args.train_hop_s))

    # build dataset windows
    wins = window_view_np(x, win=win, hop=hop)  # (W, 2, win)
    W = wins.shape[0]

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet1D(in_ch=2, base=64, tdim=64).to(device)
    ddpm = DDPM(T=args.T, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # tensorize
    data = torch.tensor(wins, device=device)  # (W, C, T)

    # training loop (epsilon prediction)
    model.train()
    for ep in range(args.epochs):
        idx = torch.randint(0, W, (args.batch,), device=device)
        x0 = data[idx]  # (B,C,T)
        x0 = zscore_per_window(x0)  # IMPORTANT: training window z-score

        t = torch.randint(0, ddpm.T, (args.batch,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = ddpm.q_sample(x0, t, noise)
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (ep + 1) % 100 == 0:
            print(f"[train] ep={ep+1}/{args.epochs} loss={loss.item():.6f}")

    # sampling: generate enough windows to cover duration_s with same hop
    model.eval()
    N = int(round(fs * args.duration_s))
    Wgen = 1 + (N - win) // hop
    if Wgen <= 0:
        raise ValueError("duration too short for windowing")

    with torch.no_grad():
        gen_w = ddpm.p_sample_loop(model, shape=(Wgen, 2, win)).cpu().numpy().astype(np.float32)

    # stitch
    y = overlap_add(gen_w, hop=hop)  # (2, N_stitched)
    y = y[:, :N]  # trim

    t = np.arange(N, dtype=np.float32) / fs
    out_df = pd.DataFrame({"time": t, "F7": y[0], "F8": y[1]})

    out_path = Path(args.out)

    # append tag to avoid overwriting
    if args.tag:
        out_path = out_path.with_name(f"{out_path.stem}_{args.tag}{out_path.suffix}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] saved diffusion-only: {out_path} (windows={Wgen}, fs={fs})")


if __name__ == "__main__":
    main()
