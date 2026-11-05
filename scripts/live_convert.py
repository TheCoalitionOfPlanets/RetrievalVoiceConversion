"""Record from the mic, convert to the trained voice, play it back — in a loop.

    uv run python scripts/live_convert.py

ENTER starts recording, ENTER stops it. The clip is converted and played back.
The model is loaded once and stays resident, so only the first take pays the
~10 s startup cost.
"""

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.conversion_service import ConversionService  # noqa: E402
from src.utils.config import load_inference_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("live")


def _recorder(path: Path, rate: int):
    if shutil.which("arecord"):
        return subprocess.Popen(
            ["arecord", "-q", "-f", "S16_LE", "-c1", "-r", str(rate), str(path)],
            stdin=subprocess.DEVNULL,
        )
    if shutil.which("ffmpeg"):
        return subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "pulse",
             "-i", "default", "-ac", "1", "-ar", str(rate), "-y", str(path)],
            stdin=subprocess.DEVNULL,
        )
    raise SystemExit("need arecord (alsa-utils) or ffmpeg")


def _play(path: Path) -> None:
    for cmd in (["aplay", "-q", str(path)], ["paplay", str(path)],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, stdin=subprocess.DEVNULL)
            return
    print("  (no player found — open the file manually)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live mic -> trained voice")
    ap.add_argument("--pitch", type=int, default=0,
                    help="Transpose in semitones (+12/-12 crosses male/female)")
    ap.add_argument("--index-rate", type=float, default=None)
    ap.add_argument("--speaker-id", type=int, default=0)
    ap.add_argument("--outdir", default="data/outputs")
    ap.add_argument("--keep-input", action="store_true",
                    help="Also keep the raw recording next to each output")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--checkpoint", default=None,
                    help="e.g. data/checkpoints/G_125.pth (default: G_latest.pth)")
    ap.add_argument("--device", default=None,
                    help="cpu/cuda (default: auto). Use cpu to test mid-training "
                         "without competing with the training process for VRAM")
    ap.add_argument("--inference-config", default="config/inference.yaml")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / ".input.wav"

    inference_config = load_inference_config(args.inference_config)
    if args.checkpoint:
        inference_config.checkpoint = args.checkpoint
    if args.device:
        inference_config.device = args.device

    print("Loading model (once)...")
    service = ConversionService(inference_config=inference_config).load()
    print("\n" + "=" * 60)
    print(f" Ready. pitch={args.pitch:+d} semitones")
    print(" ENTER = record, ENTER again = stop. 'q' = quit.")
    print(f" Outputs -> {outdir}/")
    print("=" * 60 + "\n")

    n = len(list(outdir.glob("out_*.wav")))

    while True:
        try:
            if input("ENTER to record (q to quit): ").strip().lower() == "q":
                break
        except EOFError:
            break

        proc = _recorder(tmp, args.rate)
        try:
            input("  \033[31m● RECORDING\033[0m — ENTER to stop: ")
        except EOFError:
            pass
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)

        if not tmp.exists() or tmp.stat().st_size < 4096:
            print("  nothing captured — is the mic muted?\n")
            continue

        out = outdir / f"out_{n:03d}.wav"
        t0 = time.time()
        try:
            service.convert_file(
                str(tmp), str(out),
                speaker_id=args.speaker_id,
                pitch_shift=args.pitch,
                index_rate=args.index_rate,
            )
        except Exception as exc:
            print(f"  conversion failed: {type(exc).__name__}: {exc}\n")
            continue

        print(f"  converted in {time.time()-t0:.1f}s -> {out}")
        if args.keep_input:
            shutil.copy(tmp, outdir / f"in_{n:03d}.wav")
        print("  playing...")
        _play(out)
        print()
        n += 1

    tmp.unlink(missing_ok=True)
    print(f"\nDone. {n} clips in {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
