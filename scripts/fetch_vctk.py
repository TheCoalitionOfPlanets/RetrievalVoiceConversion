"""Stream jspaulsen/vctk shard-by-shard, extract one mic per speaker, delete shard.

Peak disk stays ~500 MB above the extracted audio, so the full 17 GB corpus
never lands on disk at once.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "jspaulsen/vctk"
FILE = "data/train-{:05d}-of-00034.parquet"


def extract(shard: int, out_root: Path, mic: str, only: set[str] | None, work: Path) -> dict:
    fname = FILE.format(shard)
    local = hf_hub_download(REPO, fname, repo_type="dataset", local_dir=work)

    written, skipped, bytes_out = 0, 0, 0
    f = pq.ParquetFile(local)

    for batch in f.iter_batches(batch_size=64, columns=["id", "speaker_id", "mic_id", "audio"]):
        ids = batch.column("id").to_pylist()
        spks = batch.column("speaker_id").to_pylist()
        mics = batch.column("mic_id").to_pylist()
        auds = batch.column("audio").to_pylist()

        for uid, spk, m, a in zip(ids, spks, mics, auds):
            if m != mic or (only and spk not in only):
                skipped += 1
                continue
            d = out_root / spk
            d.mkdir(parents=True, exist_ok=True)
            # stored bytes are already 48 kHz WAV/PCM_16 -> write through, no re-encode
            p = d / f"{uid}.wav"
            p.write_bytes(a["bytes"])
            written += 1
            bytes_out += len(a["bytes"])

    del f
    os.remove(local)
    shutil.rmtree(work / ".cache", ignore_errors=True)
    return {"written": written, "skipped": skipped, "mb": round(bytes_out / 1e6, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mic", default="mic1", choices=["mic1", "mic2"])
    ap.add_argument("--shards", required=True, help="e.g. 22 or 0-33")
    ap.add_argument("--only", default=None, help="comma-separated speaker ids")
    ap.add_argument("--work", required=True)
    ap.add_argument("--state", default=None, help="json file for resume")
    args = ap.parse_args()

    if "-" in args.shards:
        a, b = args.shards.split("-")
        shards = list(range(int(a), int(b) + 1))
    else:
        shards = [int(x) for x in args.shards.split(",")]

    only = set(args.only.split(",")) if args.only else None
    out_root = Path(args.out)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state) if args.state else None
    done = set()
    if state_path and state_path.exists():
        done = set(json.loads(state_path.read_text()).get("done", []))

    total_w = 0
    for i, s in enumerate(shards):
        if s in done:
            print(f"[{i+1}/{len(shards)}] shard {s:02d} already done, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            r = extract(s, out_root, args.mic, only, work)
        except Exception as e:
            print(f"[{i+1}/{len(shards)}] shard {s:02d} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        total_w += r["written"]
        print(f"[{i+1}/{len(shards)}] shard {s:02d}: wrote {r['written']} files "
              f"({r['mb']} MB) in {time.time()-t0:.0f}s  [total {total_w}]", flush=True)
        done.add(s)
        if state_path:
            state_path.write_text(json.dumps({"done": sorted(done)}))

    print(f"DONE: {total_w} files -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
