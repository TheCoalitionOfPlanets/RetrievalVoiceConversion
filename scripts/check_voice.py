"""Sanity-check recorded target audio before committing hours to training.

    uv run python scripts/check_voice.py data/raw/myvoice
"""

import sys
from pathlib import Path

import librosa
import numpy as np


def clipped_fraction(y: np.ndarray, limit: float = 0.995, minrun: int = 3) -> float:
    """Fraction of samples sitting in a run of `minrun`+ pinned samples."""
    mask = np.abs(y) > limit
    if not mask.any():
        return 0.0
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    lengths = ends - starts
    return float(lengths[lengths >= minrun].sum()) / y.size


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/myvoice")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac"})

    if not files:
        print(f"No audio found in {root}")
        return 1

    total = 0.0
    clipped = quiet = 0
    rolloffs: list[float] = []
    worst: list[tuple[float, str]] = []

    for p in files:
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if y.size == 0:
            continue
        total += y.size / sr

        peak = float(np.abs(y).max())
        # A lone sample touching the ceiling is harmless. Real clipping is
        # flat-topping: runs of consecutive samples pinned at the limit.
        if clipped_fraction(y) > 0.0005:
            clipped += 1
            worst.append((clipped_fraction(y), p.name))
        if peak < 0.05:
            quiet += 1

        # 95% spectral rolloff on the loudest second -> real captured bandwidth
        win = min(y.size, sr)
        start = max(0, int(np.argmax(np.abs(y))) - win // 2)
        seg = y[start : start + win]
        if seg.size > 2048:
            rolloffs.append(
                float(np.percentile(librosa.feature.spectral_rolloff(
                    y=seg, sr=sr, roll_percent=0.95), 90))
            )

    band = float(np.median(rolloffs)) if rolloffs else 0.0

    print(f"files           : {len(files)}")
    print(f"total duration  : {total/60:.1f} min")
    print(f"clipped (>0.99) : {clipped}")
    print(f"too quiet(<0.05): {quiet}")
    print(f"real bandwidth  : ~{band/1000:.1f} kHz (95% rolloff)")
    print()

    ok = True
    if total < 600:
        print(f"! only {total/60:.1f} min — aim for 15+. Below ~10 min quality drops sharply.")
        ok = False
    if band < 7000:
        print(f"! bandwidth ~{band/1000:.1f} kHz. The mic is capturing a narrow band "
              "(a 16 kHz laptop DMIC does this). A 32 kHz model trained on it "
              "will sound muffled — use an external/headset mic.")
        ok = False
    elif band < 11000:
        print(f"! bandwidth ~{band/1000:.1f} kHz is on the low side for a 32 kHz model.")
    if clipped:
        print(f"! {clipped} of {len(files)} files are genuinely clipped "
              "(flat-topped peaks — permanent distortion, cannot be repaired).")
        print("  Lower the mic gain, then delete and re-record just these:")
        for frac, name in sorted(worst, reverse=True)[:15]:
            print(f"      {name}   {frac*100:.3f}% pinned")
        if len(worst) > 15:
            print(f"      ... and {len(worst)-15} more")
        print("  Gain:  pactl set-source-volume @DEFAULT_SOURCE@ 70%")
        ok = False
    if quiet > len(files) * 0.2:
        print(f"! {quiet} very quiet files — move closer or raise gain.")
        ok = False

    print("looks good — ready to preprocess." if ok else "fix the above before training.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
