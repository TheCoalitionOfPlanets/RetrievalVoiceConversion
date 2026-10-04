"""Sanity-check recorded target audio before committing hours to training.

    uv run python scripts/check_voice.py data/raw/myvoice
"""

import sys
from pathlib import Path

import librosa
import numpy as np


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/myvoice")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac"})

    if not files:
        print(f"No audio found in {root}")
        return 1

    total = 0.0
    clipped = quiet = 0
    rolloffs: list[float] = []

    for p in files:
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if y.size == 0:
            continue
        total += y.size / sr

        peak = float(np.abs(y).max())
        if peak > 0.99:
            clipped += 1
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
    if clipped > len(files) * 0.05:
        print(f"! {clipped} clipped files — lower the input gain and redo those.")
        ok = False
    if quiet > len(files) * 0.2:
        print(f"! {quiet} very quiet files — move closer or raise gain.")
        ok = False

    print("looks good — ready to preprocess." if ok else "fix the above before training.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
