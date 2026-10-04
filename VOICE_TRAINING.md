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

## Step 1 — record

```sh
./scripts/record_voice.sh                 # -> data/raw/myvoice/, 12s takes
./scripts/record_voice.sh data/raw/myvoice 15   # longer takes
```

ENTER records a take, then ENTER keeps it or `r` redoes it. Stop any time with
`q`; re-running continues where you left off. Prompts come from
`scripts/prompts.txt` (40 phonetically varied sentences — they cover a wide
phoneme range, which matters more than total minutes).

What actually determines quality:

- **15–30 minutes.** Below ~10 min the voice stays generic. Past ~30 min returns
  flatten fast.
- **One mic, one distance, one room, one session.** Consistency beats quantity.
  Varying mic distance teaches the model that your timbre is inconsistent.
- **Dead silence between phrases** — no fan, keyboard, traffic, music, or a TV
  two rooms away. The silence splitter cuts on quiet, so background noise
  silently becomes part of "your voice".
- **Use a headset or external mic, not the laptop's built-in.** `arecord -l`
  shows a `DMIC16kHz` device on this machine — a 16 kHz capture has nothing
  above 8 kHz, and a 32 kHz model trained on it sounds permanently muffled.
- **Match your intended use.** Converting singing? Record some singing. The
  model can only reproduce a pitch range it has heard.
- Speak normally. Don't perform, don't whisper, don't shout.

## Step 2 — verify before you burn GPU hours

```sh
uv run python scripts/check_voice.py data/raw/myvoice
```

Reports duration, clipping, level, and **real captured bandwidth** — that last
one catches the 16 kHz-mic trap, which is invisible until the model finishes and
sounds wrong. For reference, `data/reference/p247` scores ~19.2 kHz.

## Step 3 — preprocess, index, train

`data/processed/` currently holds **p247** features. Clear them, or the old
speaker is trained alongside yours:

```sh
rm -rf data/processed/* data/indexes/*

uv run python scripts/preprocess.py        # reads data/raw/ by default
uv run python scripts/build_index.py
uv run python scripts/train.py
```

Confirm `data/processed/speakers.json` reads `{"myvoice": 0}` and nothing else
before starting a long run.

## Step 4 — convert

```sh
uv run python scripts/infer.py --input some_source.wav --output converted.wav
```

Use `--pitch-shift` in semitones when source and target ranges differ — roughly
+12 for a male source into a female target, -12 the other way.

## Keep `data/raw/` to exactly one folder

`build_speaker_map` treats **every subdirectory as a separate speaker**. A stray
folder silently turns a single-voice fine-tune into a two-speaker run, and the
result sounds like neither. One folder in, one voice out.

## Consent

Train on your own voice, or on a voice you have explicit permission to clone.
