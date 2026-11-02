"""Shared constants, selection helpers and process utilities for the OpenNeuro pipeline.

The scientific protocol (preprocessing, generator settings, evaluation interval)
is fixed to the published study. Users choose *which* participants and seeds to
run, not *how* they are processed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Dataset and selection
# ---------------------------------------------------------------------------
DATASET_ID = "ds005385"
DATASET_VERSION = "1.0.2"
DEFAULT_BIDS_ROOT = REPO_ROOT / "data" / "openneuro" / DATASET_ID
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

SESSION = "ses-1"
TASK = "EyesClosed"
ACQUISITION = "pre"
CHANNELS = ("F7", "F8")

# The 30 participants analysed in the paper, in the order used by the paper.
PAPER_SUBJECTS = [
    "sub-012", "sub-050", "sub-062", "sub-122", "sub-140", "sub-144",
    "sub-152", "sub-157", "sub-180", "sub-305", "sub-308", "sub-320",
    "sub-323", "sub-331", "sub-349", "sub-402", "sub-407", "sub-412",
    "sub-425", "sub-449", "sub-470", "sub-518", "sub-529", "sub-542",
    "sub-566", "sub-573", "sub-574", "sub-575", "sub-597", "sub-603",
]
PAPER_SEEDS = list(range(20))

# ---------------------------------------------------------------------------
# Locked protocol (recorded in run_config.json; executed by the existing scripts)
# ---------------------------------------------------------------------------
FS = 256
DURATION_SEC = 180
PREPROCESSED_ROWS = FS * DURATION_SEC  # 46080
EVAL_START_SEC = 20
EVAL_DURATION_SEC = "139.99609375"
EVAL_DIR_SUFFIX = f"start20_dur{EVAL_DURATION_SEC}"

EVALUATE_ARGS = [
    "--ch1", CHANNELS[0], "--ch2", CHANNELS[1], "--fs", str(FS),
    "--win_s", "10", "--hop_s", "5", "--bp_low", "1", "--bp_high", "45",
    "--zscore", "global", "--dfa_min_s", "0.2", "--dfa_max_s", "1.25",
    "--start_sec", str(EVAL_START_SEC), "--duration_sec", EVAL_DURATION_SEC,
    "--plots", "--no_gc",
]

WGAN_ARGS = [
    "--fs", str(FS), "--duration_s", str(DURATION_SEC),
    "--train_win_s", "10", "--train_hop_s", "5",
    "--gen_steps", "3000", "--batch", "16", "--n_critic", "3", "--gp_lambda", "50",
    "--latent", "100", "--lr", "1e-4", "--gen_norm", "group",
    "--diag_n_gen", "200", "--diag_no_save_windows", "--save_180s",
]

PROTOCOL = {
    "dataset": f"OpenNeuro {DATASET_ID} v{DATASET_VERSION}",
    "recording": {"session": SESSION, "task": TASK, "acquisition": ACQUISITION,
                  "channels": list(CHANNELS)},
    "preprocessing": {"target_fs": FS, "duration_sec": DURATION_SEC,
                      "operations": ["channel selection", "resampling", "first-180-s crop", "V-to-uV"]},
    "evaluation": {"start_sec": EVAL_START_SEC, "duration_sec": float(EVAL_DURATION_SEC),
                   "window_sec": 10.0, "hop_sec": 5.0, "bandpass_hz": [1.0, 45.0],
                   "zscore": "global", "dfa_scale_sec": [0.2, 1.25]},
    "nmm": {"coupling": 0.02, "duration_sec": 180.0},
    "diffusion_only": {"T": 1000, "optimization_updates": 800, "batch_size": 8, "learning_rate": 1e-3,
                       "train_window_sec": 10.0, "train_hop_sec": 5.0},
    "add_residual_diffusion": {"T": 1000, "optimization_updates": 800, "batch_size": 8,
                               "learning_rate": 1e-3, "std_match": True, "mixing_coefficient_b": 1.0},
    "wgan_gp": {"n_critic": 3, "gp_lambda": 50, "generator_updates": 3000, "batch_size": 16,
                "latent": 100, "learning_rate": 1e-4, "adam_betas": [0.0, 0.9], "gen_norm": "group",
                "train_window_sec": 10.0, "train_hop_sec": 5.0},
    "training_data": "Each generator is trained on the same participant recording that is evaluated "
                     "(in-sample temporal fidelity, as stated in the paper).",
}

METRICS = [
    "psd_loglog_slope_all_2_40hz",
    "dfa_x_alpha",
    "acf_x_abs_0_1s",
    "acf_x_abs_0_5s",
    "acf_x_abs_1_0s",
]
METRIC_DISPLAY = {
    "psd_loglog_slope_all_2_40hz": "PSD slope (2–40 Hz)",
    "dfa_x_alpha": "DFA α",
    "acf_x_abs_0_1s": "ACF ≤ 0.1 s",
    "acf_x_abs_0_5s": "ACF ≤ 0.5 s",
    "acf_x_abs_1_0s": "ACF ≤ 1.0 s",
}
CORE_MODELS = ["NMM-only", "Diffusion-only", "Add"]
ALL_MODELS = [*CORE_MODELS, "WGAN-GP"]


def eval_dirname(model: str, seed: int) -> str:
    """Evaluation sub-directory names inside subjects/<sub>/seed<N>/ (fixed by the runners)."""
    return {
        "Real": f"real_{EVAL_DIR_SUFFIX}",
        "NMM-only": f"nmm_seed{seed}_{EVAL_DIR_SUFFIX}",
        "Diffusion-only": f"diffusion_only_{EVAL_DIR_SUFFIX}",
        "Add": f"add_stdmatched_b1.0_{EVAL_DIR_SUFFIX}",
        "WGAN-GP": f"gan_{EVAL_DIR_SUFFIX}",
    }[model]


def edf_relpath(subject: str) -> str:
    return (f"{subject}/{SESSION}/eeg/"
            f"{subject}_{SESSION}_task-{TASK}_acq-{ACQUISITION}_eeg.edf")


def preprocessed_name(subject: str) -> str:
    return f"{subject}_{SESSION}_{TASK}_acq-{ACQUISITION}_180s_256Hz.csv"


# ---------------------------------------------------------------------------
# Participant / seed selection shared by every command
# ---------------------------------------------------------------------------
def add_selection_args(parser) -> None:
    group = parser.add_argument_group("participant and seed selection")
    sub = group.add_mutually_exclusive_group()
    sub.add_argument("--n-subjects", type=int, metavar="N",
                     help="Use the first N of the paper's 30 participants (paper order). Default: 30.")
    sub.add_argument("--subjects", nargs="+", metavar="sub-XXX",
                     help="Explicit participant labels, e.g. --subjects sub-012 sub-407.")
    seed = group.add_mutually_exclusive_group()
    seed.add_argument("--n-seeds", type=int, metavar="N",
                      help="Use seeds 0..N-1. Default: 20 (seeds 0-19, as in the paper).")
    seed.add_argument("--seeds", nargs="+", type=int, metavar="S",
                      help="Explicit seed numbers, e.g. --seeds 0 1 2.")


def resolve_selection(args) -> tuple[list[str], list[int]]:
    if args.subjects:
        subjects = list(dict.fromkeys(args.subjects))
        bad = [s for s in subjects if not (s.startswith("sub-") and s[4:].isdigit())]
        if bad:
            raise SystemExit(f"[ERROR] Invalid participant label(s): {bad}. Use the BIDS form sub-012.")
    else:
        n = 30 if args.n_subjects is None else args.n_subjects
        if not 1 <= n <= len(PAPER_SUBJECTS):
            raise SystemExit(f"[ERROR] --n-subjects must be between 1 and {len(PAPER_SUBJECTS)}.")
        subjects = PAPER_SUBJECTS[:n]
    if args.seeds is not None:
        seeds = sorted(dict.fromkeys(args.seeds))
        if any(s < 0 for s in seeds):
            raise SystemExit("[ERROR] Seeds must be non-negative integers.")
    else:
        n = 20 if args.n_seeds is None else args.n_seeds
        if n < 1:
            raise SystemExit("[ERROR] --n-seeds must be at least 1.")
        seeds = list(range(n))
    return subjects, seeds


# ---------------------------------------------------------------------------
# Reporting and subprocess helpers
# ---------------------------------------------------------------------------
@dataclass
class Report:
    """Collects check results as ERROR / WARN / OK lines."""
    title: str
    items: list[dict] = field(default_factory=list)

    def add(self, level: str, check: str, detail: str, hint: str = "") -> None:
        self.items.append({"level": level, "check": check, "detail": detail, "hint": hint})

    def ok(self, check: str, detail: str = "") -> None:
        self.add("OK", check, detail)

    def warn(self, check: str, detail: str, hint: str = "") -> None:
        self.add("WARN", check, detail, hint)

    def error(self, check: str, detail: str, hint: str = "") -> None:
        self.add("ERROR", check, detail, hint)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.items if i["level"] == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.items if i["level"] == "WARN")

    def print(self, verbose_ok: bool = False) -> None:
        print(f"\n=== {self.title} ===")
        for item in self.items:
            if item["level"] == "OK" and not verbose_ok:
                continue
            line = f"[{item['level']:5s}] {item['check']}"
            if item["detail"]:
                line += f": {item['detail']}"
            print(line)
            if item["hint"]:
                print(f"        -> {item['hint']}")
        n_ok = sum(1 for i in self.items if i["level"] == "OK")
        print(f"--- {n_ok} OK, {self.n_warnings} warning(s), {self.n_errors} error(s)")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"title": self.title, "created_at": now(), "items": self.items},
                                   indent=2, ensure_ascii=False), encoding="utf-8")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


class StepFailed(RuntimeError):
    pass


def run_logged(cmd: list[str], log_path: Path, env: dict | None = None, cwd: Path = REPO_ROOT) -> float:
    """Run a command, write its full output to log_path, raise StepFailed on a non-zero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env.setdefault("MPLBACKEND", "Agg")
    full_env["PYTHONUNBUFFERED"] = "1"  # logs show progress while a step is running
    if env:
        full_env.update(env)
    start = time.time()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# {now()}  $ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=full_env, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise StepFailed(
            f"Command failed (exit {proc.returncode}). Last lines of {log_path}:\n    "
            + "\n    ".join(tail)
        )
    return elapsed


def python_exe() -> str:
    return sys.executable


def rel(path: Path) -> str:
    """Repository-relative path for reports (never an absolute home-directory path)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
