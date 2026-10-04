# Training the model on your own voice

## How RVC decides whose voice comes out

RVC does **not** learn "speaker A → speaker B". It learns *one* voice — the one
in `data/raw/` — and learns to render arbitrary speech content in that voice.

```
   any source audio              trained voice
  (you, a friend, a song)   ->   comes out here
          |                            ^
          |  HuBERT: what was said     |
          |  RMVPE/harvest: the pitch  |
          +----------------------------+
```

At inference the source only supplies **content and pitch**. Timbre — the thing
that makes a voice recognisably yours — comes entirely from training.

**So: whatever is in `data/raw/` is what the app outputs.** `p247` was a VCTK
stand-in used to prove the pipeline. It now lives in `data/reference/p247/`,
out of the training root, so it cannot sneak into your run.

## You are fine-tuning, not training from scratch

`data/pretrained/pretrained_v2/` holds the official RVC v2 32 kHz base
(`f0G32k.pth`, `f0D32k.pth`), already wired into `config/train.yaml`. The base
contributes general speech structure; your recordings only need to supply
*your timbre*. That is why ~15 minutes is enough rather than many hours, and why
this is feasible on a 4 GB card at all.

## The whole sequence, start to finish

```sh
# 0. plug in a headset or external mic (NOT the laptop's built-in — see below)

# 1. record. ENTER starts a take, ENTER stops it, r redoes it, q quits.
#    Resumable: re-run any time to continue where you left off.
./scripts/record_voice.sh

# 2. verify BEFORE spending GPU time. Catches a narrow-band mic, clipping,
#    levels that are too low, and not enough material.
uv run python scripts/check_voice.py data/raw/myvoice

# 3. clear the old p247 artifacts, or they get trained alongside you
rm -rf data/processed/* data/indexes/* data/checkpoints/*

# 4. confirm exactly one speaker will be picked up
ls data/raw/            # must print only: myvoice

# 5. extract HuBERT content + F0      (~5-10 min on CPU/GPU for 15 min audio)
uv run python scripts/preprocess.py

# 6. sanity-check the speaker map before the long run
cat data/processed/speakers.json      # must read exactly {"myvoice": 0}

# 7. build the FAISS retrieval index   (fast, under a minute)
uv run python scripts/build_index.py

# 8. fine-tune from the 32 kHz pretrained base   (the long one)
uv run python scripts/train.py

# 9. convert something into your voice
uv run python scripts/infer.py --input source.wav --output converted.wav --pitch 0
```

Steps 5-8 can be run as one command once you trust them:

```sh
uv run python scripts/run_pipeline.py
```

Watch training in a second terminal:

```sh
uv run tensorboard --logdir data/logs
```

Training resumes from the newest checkpoint in `data/checkpoints/`, so Ctrl-C
is safe — you lose at most the epochs since the last save (`save_every_epoch: 5`
in `config/train.yaml`).

## Step 1 in detail — recording

`./scripts/record_voice.sh` prints a paragraph, you press ENTER, read it aloud,
press ENTER again to stop. It tracks a running total against a 15-minute goal
and shows a progress bar.

```sh
./scripts/record_voice.sh                      # default: data/raw/myvoice
GOAL_MIN=20 ./scripts/record_voice.sh          # aim for 20 minutes instead
PROMPTS=my_own_text.txt ./scripts/record_voice.sh   # your own paragraphs
```

Prompts live in `scripts/prompts.txt` — 20 paragraphs, blank-line separated,
chosen to spread across a wide phoneme range (plus numbers, questions, and
tongue-twisters). The list cycles once exhausted; **re-reading the same
paragraph is fine.** The model learns timbre, not text.

What actually determines quality:

- **15-30 minutes.** Below ~10 min the voice stays generic. Past ~30 min returns
  flatten fast.
- **One mic, one distance, one room, one session.** Consistency beats quantity.
  Varying mic distance teaches the model your timbre is inconsistent.
- **Dead silence between phrases** — no fan, keyboard, traffic, music, or a TV
  two rooms away. The splitter cuts on quiet, so background noise silently
  becomes part of "your voice".
- **Not the laptop's built-in mic.** `arecord -l` shows a `DMIC16kHz` device on
  this machine. A 16 kHz capture has nothing above 8 kHz, and a 32 kHz model
  trained on it sounds permanently muffled no matter how long you train.
- **Match your intended use.** Converting singing? Record singing. The model
  cannot reproduce a pitch range it has never heard.
- Speak normally. Don't perform, don't whisper, don't shout.

## Step 2 in detail — verifying

```sh
uv run python scripts/check_voice.py data/raw/myvoice
```

Reports duration, clipping, levels, and **real captured bandwidth**. That last
number is the one that matters — it catches the narrow-mic trap, which is
otherwise invisible until training finishes and the output sounds wrong.
`data/reference/p247` scores ~19.2 kHz; anything under ~7 kHz is a bad mic.

Fix whatever it flags and re-run before moving on.

## Using it: live mic -> your voice

```sh
uv run python scripts/live_convert.py
```

ENTER records, ENTER stops, the clip is converted and played back immediately.
The model loads **once** and stays resident, so only the first take pays the
~10 s startup; after that conversion runs at roughly real time (RTF ~0.8 on the
3050).

```sh
uv run python scripts/live_convert.py --pitch 0          # same pitch
uv run python scripts/live_convert.py --pitch -12        # drop an octave
uv run python scripts/live_convert.py --keep-input       # also save the raw take
uv run python scripts/live_convert.py --outdir /tmp/vc   # somewhere else
```

Outputs land in `data/outputs/out_000.wav`, `out_001.wav`, ... and numbering
continues across runs.

### Converting an existing file instead

`scripts/infer.py` takes a real path — `source.wav` in the examples is a
placeholder, not a file that exists:

```sh
uv run python scripts/infer.py \
  --input data/reference/p247/p247_001_mic1.wav \
  --output data/outputs/demo.wav \
  --pitch 0
```

`--input` also accepts a directory, in which case `--output` is treated as a
directory too.

### Dialling in the result

| flag | effect |
|---|---|
| `--pitch N` | transpose N semitones. Use when the source voice sits in a different range than yours; ±12 is an octave |
| `--index-rate R` | 0 = ignore the retrieval index, 1 = lean on it hard. Default 0.75. Lower it if output sounds smeared, raise it if timbre drifts off |
| `--f0-method` | `harvest` (default, robust), `dio` (faster), `crepe` (needs the `crepe` extra) |
| `--checkpoint` | pick a specific epoch, e.g. `data/checkpoints/G_250.pth`, instead of `G_latest.pth` |

## If something goes wrong

| symptom | cause |
|---|---|
| CUDA out of memory | lower `batch_size` or `segment_size` in `config/train.yaml` (4 GB card) |
| `speakers.json` has 2+ entries | an extra folder in `data/raw/` — remove it, re-run preprocess |
| output sounds muffled | narrow-band mic; check step 2 |
| output sounds like p247 | stale `data/processed/` — you skipped step 3 |
| output pitch is wrong | use `--pitch` semitones at step 9 (±12 crosses male/female) |
| "No usable samples" | clips too short, or `data/raw/` empty |
| `AudioLoadError: File not found: source.wav` | `source.wav` is a placeholder — pass a real path, or use `live_convert.py` |

## Keep `data/raw/` to exactly one folder

`build_speaker_map` treats **every subdirectory as a separate speaker**. A stray
folder silently turns a single-voice fine-tune into a two-speaker run, and the
result sounds like neither. One folder in, one voice out.

## Consent

Train on your own voice, or on a voice you have explicit permission to clone.
