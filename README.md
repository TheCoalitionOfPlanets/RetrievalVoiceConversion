# RVC — Retrieval-based Voice Conversion

A from-scratch RVC v2 implementation: HuBERT content features + F0 conditioning
+ FAISS retrieval + a VITS/NSF-HiFiGAN synthesizer, trained adversarially.

Tuned for a **single 4 GB GPU**. Weights are byte-compatible with the official
RVC v2 32 kHz pretrained models, so a usable voice needs ~15 minutes of audio
rather than tens of hours.

---

## How it works

```
                 ┌──────────── training only ─────────────┐
                 │                                        │
  audio ──┬─► linear spectrogram ─► PosteriorEncoder ─► z ─┼─► flow ─► z_p
          │                                            │   │           │
          │                                            │   │      KL divergence
          │                                     random slice            │
          │                                            │               │
          ├─► 16 kHz ─► HuBERT ─► content (768-d, 50 fps)               │
          │                           │                                 │
          │                      ×2 repeat → 100 fps                    │
          │                           │                                 │
          │                     FAISS retrieval  (inference only)       │
          │                           │                                 │
          │                      TextEncoder ◄── coarse F0 (256 bins) ──┘
          │                           │
          └─► F0 (rmvpe) ─────────────┴──► prior p(z | content, f0)
                   │
                   └──────────► NSF source ─► GeneratorNSF ─► waveform
                                                   │
                                      MultiPeriodDiscriminator
```

At **training** time the posterior encoder sees the real spectrogram and the
flow maps it toward the prior built from content + pitch; the decoder is trained
on a random 40-frame slice of the latent. At **inference** there is no
spectrogram, so the prior is sampled directly and the flow is run in reverse.

Retrieval is the "R" in RVC: each content frame is replaced by a
distance-weighted blend of its nearest neighbours from the target speaker's own
feature bank, which pulls timbre toward the target and suppresses artefacts the
decoder would otherwise invent for unfamiliar phonetics.

### Layout

| Path | Role |
|---|---|
| `src/models/vits/` | `encoder` (prior/posterior), `flow`, `decoder` (NSF-HiFiGAN), `discriminator`, `model` (synthesizer), plus `modules`/`attentions`/`commons` |
| `src/feature_extraction/` | `hubert` (content), `rmvpe` (F0 + coarse quantisation), `spectrogram`, `extractor` (orchestrator) |
| `src/retrieval/` | `faiss_index` (build/search), `retrieval_service` (blending) |
| `src/training/` | `dataset` (+ collate, bucket sampler), `losses`, `callbacks`, `trainer` |
| `src/inference/` | `converter` (chunking, pitch shift, protect, RMS match), `voice_conversion_pipeline` |
| `src/services/` | `feature_service` (preprocess), `training_service`, `conversion_service`, `model_registry`, `pipeline_service` |
| `src/utils/` | `config`/`settings` (typed), `audio`, `checkpoint`, `device`, `logger`, `seed` |
| `config/` | `model.yaml`, `train.yaml`, `inference.yaml` |

---

## The dataset you need

**One speaker — the voice you want to convert *to*.** Nothing else. No
transcripts, no labels, no paired source/target recordings. RVC learns a
speaker's timbre, not a language, because HuBERT already supplies the phonetics.

### How much

| Amount | Result |
|---|---|
| < 5 min | Recognisable but unstable; artefacts on unfamiliar phonetics |
| **10–20 min** | **The sweet spot with the pretrained base — this is what to aim for** |
| 30–60 min | Modest gains, mostly in robustness |
| > 2 h | Diminishing returns unless you train from scratch |

Quality matters far more than quantity past ~15 minutes. Twenty clean minutes
beats two noisy hours every time.

### What it must be

- **Clean and dry.** No background music, no reverb, no second speaker. Reverb
  in the training set gets baked permanently into the model.
- **One consistent mic and room.** Mixing sources teaches the model to average
  two channels.
- **No clipping**, no aggressive compression or de-noising artefacts.
- **≥ 32 kHz source.** Anything lower has no real content above its Nyquist
  limit and the vocoder will learn a muffled band. 44.1/48 kHz is ideal;
  it gets resampled to 32 kHz.
- **Phonetically varied.** All vowels and consonants. A speaker reading varied
  sentences is ideal.
- **Pitch range that matches your use.** A model trained only on flat speech
  will not sing. If you want singing, include singing.

Mono/stereo, WAV/FLAC/MP3, any clip length — all handled. Long files are cut at
silences into ≤ 5 s chunks automatically.

If your source has music behind it, separate the vocals first (UVR5 or Demucs)
and feed only the vocal stem.

### Layout

```
data/raw/<speaker-name>/*.wav         # any nesting; folder name = speaker
```

A single folder of files directly under the root also works and is treated as
one speaker.

### What is already here

The training root `data/raw/` is where **your** voice goes — see
**`VOICE_TRAINING.md`** for the recording workflow (`scripts/record_voice.sh`,
`scripts/check_voice.py`) and what makes a take usable.

A VCTK speaker (`p247`, 475 utterances, 15.6 min, 48 kHz) sits in
`data/reference/p247/` — deliberately *outside* the training root so it cannot
leak into a run. It was the stand-in used to prove this pipeline end to end, and
it doubles as clean source audio to convert once you have a model.
`data/DATASET.md` documents its provenance.

Remember what the split means: whatever is under `data/raw/` is the voice the
model *becomes*; anything you pass to `scripts/infer.py` only supplies content
and pitch.

---

## Setup

```sh
uv sync
```

Pretrained base models (already present in `data/pretrained/`; re-fetch with):

```sh
uv run hf download lj1995/VoiceConversionWebUI \
  pretrained_v2/f0G32k.pth pretrained_v2/f0D32k.pth \
  --local-dir data/pretrained
```

These load at **100 % tensor match** into this architecture. Training without
them works but needs an order of magnitude more audio.

---

## Commands

### 1. Preprocess — slice, extract HuBERT features and F0

```sh
uv run python scripts/preprocess.py --raw data/raw --device cuda
```

Writes `data/processed/{wavs,features,f0,f0_coarse}/`, `filelist.txt` and
`speakers.json`. Budget ~400 MB per 20 minutes of audio.

### 2. Build the retrieval index

```sh
uv run python scripts/build_index.py
```

### 3. Train

```sh
uv run python scripts/train.py --device cuda
```

Resumes automatically from the newest checkpoint in `data/checkpoints/`, so
Ctrl-C is safe. Monitor with:

```sh
uv run tensorboard --logdir data/logs
```

Useful overrides: `--epochs N`, `--batch-size N`, `--no-fp16`.

### 4. Convert

```sh
uv run python scripts/infer.py \
  --input input.wav --output converted.wav \
  --pitch 0 --device cuda
```

`--input` also accepts a directory. `--pitch` transposes in semitones (+12 for
male→female, −12 the other way). `--index-rate 0..1` trades timbre strength
against clarity.

### 5. Text-to-voice (no source recording needed)

`scripts/infer.py` needs a source *audio file* to supply content — there's
no raw-audio shortcut if you just want to hear the trained voice say
something typed. `scripts/tts_infer.py` covers that case: it synthesizes the
text with Microsoft's neural TTS (via the free `edge-tts` client — the same
voice models behind Edge's Read Aloud, not the paid Azure Speech API) and
immediately feeds that synthetic audio through the trained converter as the
source. The TTS step only supplies content and prosody to convert, same as
any other `--input` file; it has no influence on the trained voice itself.

```sh
uv run python scripts/tts_infer.py \
  --text "Hello there" --output out.wav --voice en-US-AndrewNeural
```

`scripts/live_chat.py` wraps the same pipeline in a Gradio chat UI that keeps
the model loaded between turns (`uv run python scripts/live_chat.py --port 7866`).

### Everything at once

```sh
uv run python scripts/run_pipeline.py --raw data/raw --device cuda
```

### Tests

```sh
uv run pytest test/ -q        # 44 tests, no network, ~5 s
```

---

## What to expect on a 4 GB GPU

Measured on an RTX 3050 Laptop (3.7 GB usable), batch 4, fp16:

| | |
|---|---|
| Peak VRAM | **2.8 GB** typical, **2.9 GB** at the 500-frame ceiling |
| Step time | ~0.4 s |
| Generator / discriminator | 36.4 M / 55.0 M params |
| Inference | RTF ~0.2 (≈5× faster than realtime) |

For 15 minutes of audio (~250 segments, ~62 steps/epoch) that is roughly
**25 s/epoch**. Expect a recognisable voice by ~100 epochs (~45 min) and a good
one by 300–500 epochs (**2.5–4 hours**).

Watch `loss/g/mel` in TensorBoard — it should fall steadily. The adversarial
losses oscillate; that is normal and not a failure signal.

### If you hit OOM

In `config/train.yaml`, in order of preference: lower `data.max_frames` (500 →
300), then `training.batch_size` (4 → 2), then `training.segment_size`
(12800 → 8960). Keep `fp16: true`.

---

## Configuration notes

`config/model.yaml` is pinned to the RVC v2 32 kHz profile. **Do not change
`upsample_rates`, `upsample_kernel_sizes`, `gin_channels` or `spk_embed_dim`** —
those exact values are what make the pretrained weights load. 32 kHz was chosen
over 40/48 kHz because the decoder and discriminator dominate VRAM, and 32 kHz
is what keeps a full GAN step inside 4 GB.

Checkpoints embed the config they were trained with, and inference prefers it
over the YAML, so a checkpoint stays loadable after you edit configs.

F0 defaults to `rmvpe` (neural, DeepUnet+BiGRU over a mel spectrogram —
accurate on expressive/noisy speech, needs `data/pretrained/rmvpe.pt` and a
GPU for real-time use). `harvest` (pyworld, classical, no extra weights) and
`dio` (faster, less accurate) are available with `--f0-method`; `crepe` needs
`uv sync --extra crepe`.
