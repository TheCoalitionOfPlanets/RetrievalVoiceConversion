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
data/vctk/<speaker>/<speaker>_<utt>_mic1.wav   # full corpus, staging
data/raw/p225/<utt>.wav                        # the fine-tune target
```

`data/raw/` is the root `scripts/preprocess.py` reads by default, and it holds
**only the target speaker**. `data/vctk/` is staging — it is deliberately
outside `data/raw/` so a preprocess run does not pick up all 109 speakers.

## Target speaker

`p225` — female, 23, Southern England. The conventional VCTK target in voice
conversion work, and it lives entirely within one parquet shard, so the full
voice is cheap to re-fetch.

To swap targets, hardlink a different speaker (no copy, no extra disk):

```sh
rm -rf data/raw/* && mkdir -p data/raw/p226
ln data/vctk/p226/*.wav data/raw/p226/
```

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
