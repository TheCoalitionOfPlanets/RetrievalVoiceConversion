# Dataset: CSTR VCTK Corpus (0.92) — speaker p247

A single-speaker slice, sized for a disk-constrained laptop. **87 MB on disk.**

## What is here

```
data/raw/p247/p247_<utt>_mic1.wav   # 475 utterances, 15.6 min, 48 kHz
```

`data/raw/` is the default root for `scripts/preprocess.py`, so no flags needed:

```sh
uv run python scripts/preprocess.py
```

| | |
|---|---|
| **Speaker** | `p247` (demographics per the VCTK `speaker-info.txt` table, not shipped in this mirror) |
| **Utterances** | 475 (complete; p247 is not split across shards) |
| **Duration** | 15.6 min, avg 1.97 s/utt |
| **Audio** | 48 kHz, WAV / PCM_16, mono → resampled to 32 kHz by the loader |
| **License** | CC BY 4.0 |

p247 was chosen over the conventional `p225` because p225 holds only 231
utterances / 9.3 min. RVC fine-tunes are generally happier with 15+ minutes, and
both cost the same single shard to fetch.

## Source

[`jspaulsen/vctk`](https://huggingface.co/datasets/jspaulsen/vctk) — a repackaging
of the [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443) with
leading/trailing silence removed via Silero VAD (25 ms padding retained).

The other mirrors were rejected for concrete reasons:

- `CSTR-Edinburgh/vctk` — ships only a loading script, which modern `datasets`
  will not execute, and this project does not depend on `datasets` at all.
- `saeedzou/vctk-48khz` — gated, requires an authenticated account.
- `arampacha/vctk_22kHz` — 22 kHz. Upsampling to the 32 kHz target in
  `config/model.yaml` yields no real content above 11 kHz, so the vocoder would
  learn a muffled band.

## Why mic1 only

VCTK 0.92 recorded every utterance twice — an omni headset (`mic1`) and a
cardioid standing mic (`mic2`). Only `mic1` was kept: it is the closer, drier
signal, and keeping both would duplicate every utterance in the FAISS retrieval
index, where nearest-neighbour statistics would be dominated by exact-content
twins differing only in channel.

## Disk

Preprocessing writes **uncompressed** `.npy`. For this 15.6 min speaker expect
roughly 600-700 MB under `data/processed/` (768-dim HuBERT features at 100
frames/sec, plus waveform arrays). That is the intended scale.

For reference, the full 44 h / 109-speaker corpus would expand to roughly 50 GB
of features plus ~20 GB of waveforms — do not preprocess the whole corpus on
this machine.

## Fetching a different or additional speaker

`scripts/fetch_vctk.py` streams one parquet shard at a time, extracts the
requested speakers, and deletes the shard before moving on, so peak disk stays
~500 MB above the extracted audio instead of the 17 GB the full repo needs.

`data/speaker_shards.json` maps each of the 34 shards to the speakers it holds.
Look up the shard for the speaker you want, then:

```sh
# one-off venv: the script needs pyarrow, which the project does not depend on
uv venv /tmp/vctk-env --python 3.12
uv pip install --python /tmp/vctk-env/bin/python pyarrow huggingface_hub

/tmp/vctk-env/bin/python scripts/fetch_vctk.py \
  --out data/raw --mic mic1 --shards 22 --only p225 \
  --work /tmp/vctk-work --state /tmp/vctk-work/state.json
```

Each shard is ~482 MB and takes 1-2 min. The fetch is resumable via `--state`.

**Keep `data/raw/` to one speaker folder** — `build_speaker_map` treats every
subdirectory as a separate speaker, so a stray folder silently turns a
single-target fine-tune into a two-speaker run.
