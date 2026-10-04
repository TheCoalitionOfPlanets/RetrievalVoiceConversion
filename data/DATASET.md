# Dataset: CSTR VCTK Corpus (0.92)

## Source

Pulled from the Hugging Face mirror [`jspaulsen/vctk`](https://huggingface.co/datasets/jspaulsen/vctk),
a repackaging of the original [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443)
with leading/trailing silence removed via Silero VAD (25 ms padding retained on
each end).

The canonical `CSTR-Edinburgh/vctk` repo was not usable: it ships only a loading
script, which modern `datasets` no longer executes, and this project does not
depend on `datasets` at all. `saeedzou/vctk-48khz` is gated. `arampacha/vctk_22kHz`
is 22 kHz, which would have to be upsampled to reach the 32 kHz target in
`config/model.yaml` — no real content above 11 kHz, so the vocoder would learn a
muffled band. The `jspaulsen` mirror is open, plain parquet, and 48 kHz.

- **Audio**: 48 kHz, WAV / PCM_16, mono
- **Speakers**: 109 (`p225`–`p376`, plus `s5`)
- **License**: CC BY 4.0 (Public Domain dedication on the original corpus)
- **Mics**: the mirror carries both `mic1` and `mic2` for every utterance.
  Only **`mic1`** was extracted — see below.

## Why mic1 only

VCTK 0.92 recorded every utterance twice, on an omni headset (`mic1`) and a
cardioid standing mic (`mic2`). Keeping both would have meant:

- Every utterance duplicated, which skews the FAISS retrieval index — nearest-
  neighbour statistics get dominated by exact-content twins that differ only in
  channel.
- Two different room/channel characteristics in one speaker's training set,
  which a single-target fine-tune should not have to model.

`mic1` is the closer, drier signal, so that is what was kept. This also halves
the on-disk footprint.

## Layout

```
data/vctk/<speaker>/<speaker>_<utt>_mic1.wav   # full corpus, staging (source of truth)
data/target/p225/<utt>.wav                     # the fine-tune target (hardlinks)
```

`data/target/` holds **only the target speaker** and is the root to preprocess:

```sh
uv run python scripts/preprocess.py --raw data/target
```

**Why not `data/raw/`?** That is the default root, but it is currently occupied
by a `testvoice/` smoke-test fixture belonging to another working session. Two
speaker folders under one root make `build_speaker_map` emit a two-speaker map,
which would mix synthetic test clips into a real training run. `data/target/`
keeps the two apart. If `testvoice/` is gone and you want the default path back:

```sh
mkdir -p data/raw/p225 && ln -f data/vctk/p225/*.wav data/raw/p225/
```

`data/vctk/` is staging, deliberately outside both, so no preprocess run picks
up all 109 speakers by accident.

Every file under `data/target/` is a **hardlink** into `data/vctk/`, so the
target costs no extra disk and switching targets is free.

## Target speaker

`p225` — female, 23, Southern England. The conventional VCTK target in voice
conversion work, and it lives entirely within one parquet shard, so the full
voice is cheap to re-fetch.

To swap targets, hardlink a different speaker (no copy, no extra disk):

```sh
rm -rf data/target/* && mkdir -p data/target/p247
ln -f data/vctk/p247/*.wav data/target/p247/
```

Durations of the speakers retrieved so far (`mic1` only):

| speaker | utts | duration | complete? |
|---|---|---|---|
| p247 | 475 | 15.6 min | yes |
| p318 | 422 | 14.3 min | yes |
| p251 | 364 | 13.0 min | yes |
| p334 | 423 | 12.9 min | yes |
| p360 | 424 | 12.0 min | yes |
| **p225** | **231** | **9.3 min** | **yes** |

p225 is the conventional target but is one of the shorter speakers. RVC
fine-tunes are usually happier with 15+ minutes, so **p247 is the better choice
if voice identity matters more than matching published VCTK results.**

## Disk warning

Do **not** point `scripts/preprocess.py` at `data/vctk/`. Preprocessing writes
uncompressed `.npy` — at 768-dim HuBERT features, 100 frames/sec, the full 44 h
corpus would expand to roughly 50 GB of features plus ~20 GB of waveform arrays.
One speaker (~20 min) is about 400 MB, which is the intended scale here.

## Reproducing

The fetch streams one parquet shard at a time, extracts `mic1`, and deletes the
shard before moving on, so peak disk stays ~500 MB above the extracted audio
rather than the 17 GB the full repo would otherwise need. Script and the
speaker→shard map are in the session scratchpad (`fetch_vctk.py`,
`speaker_shards.json`); the fetch is resumable via `fetch_state.json`.
