#!/usr/bin/env python3
"""Check rendered labels after rebuilding figures; no EEG/training required.

Historical source_data labels and BIDS/internal subject keys are intentionally
not rewritten. Checks target visible figure/table outputs and both Fig. 5 paths.
"""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_svg(path, participant_axis=None):
    text = path.read_text(encoding="utf-8")
    assert "DFA alpha" not in text, f"Spelled-out alpha in figure: {path}"
    assert (r"DFA $\alpha$" in text or "DFA α" in text), f"Missing DFA alpha symbol: {path}"
    for threshold in ("0.1", "0.5", "1.0"):
        assert f"ACF ≤ {threshold} s" in text, f"Missing ACF range label: {path}"
    assert "2–40 Hz" in text, f"PSD range typography differs: {path}"
    assert "OpenNeuro subject" not in text and "Number of subjects" not in text, path
    if participant_axis:
        assert participant_axis in text, f"Missing participant axis: {path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-figure5-dir", type=Path)
    args = parser.parse_args()
    plots = sorted((ROOT / "figures").rglob("*.svg"))
    assert len(plots) == 9, f"Expected nine generated panels, found {len(plots)}"
    for path in plots:
        axis = ("OpenNeuro participant" if path.name.startswith("Fig5a") else
                "Number of participants" if path.name.startswith("Fig5b") else None)
        check_svg(path, axis)
    if args.extra_figure5_dir:
        for prefix, axis in [("Fig5a", "OpenNeuro participant"), ("Fig5b", "Number of participants")]:
            matches = list(args.extra_figure5_dir.glob(prefix + "*.svg"))
            assert len(matches) == 1, matches
            check_svg(matches[0], axis)
    for path in (ROOT / "tables").glob("*"):
        if path.suffix not in (".csv", ".md"):
            continue
        text = path.read_text(encoding="utf-8")
        for stale in ("DFA alpha", "30 subjects", "per subject", "cross-subject"):
            assert stale not in text, f"Stale display wording {stale!r}: {path}"
    # Check both drawing routes, including the non-default participant selection.
    for name in ("reproducibility/openneuro_figure5_core.py", "pipeline/openneuro_results.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not any("OpenNeuro subject" in s or "Number of subjects" in s for s in strings), name
    print("[PASS] Nine regenerated figure panels, table displays, and both Figure 5 drawing routes.")
    print("Figure 1 is a fixed schematic and requires separate visual inspection.")


if __name__ == "__main__":
    main()
