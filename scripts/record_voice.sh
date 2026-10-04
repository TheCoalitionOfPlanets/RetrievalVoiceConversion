#!/usr/bin/env bash
# Record training audio for an RVC target voice.
#
#   ./scripts/record_voice.sh [outdir] [seconds-per-clip]
#
# Records mono 48 kHz WAV (the loader resamples to 32 kHz). ENTER starts a take,
# the take auto-stops, then you keep or redo it. Ctrl-C to stop for the session;
# re-running continues from the next free clip number.

set -euo pipefail

OUT="${1:-data/raw/myvoice}"
SECS="${2:-12}"
PROMPTS="${PROMPTS:-scripts/prompts.txt}"

mkdir -p "$OUT"

if command -v arecord >/dev/null; then
    rec() { arecord -q -f S16_LE -c1 -r48000 -d "$SECS" "$1"; }
elif command -v ffmpeg >/dev/null; then
    rec() { ffmpeg -hide_banner -loglevel error -f pulse -i default \
                   -ac 1 -ar 48000 -t "$SECS" -y "$1"; }
else
    echo "need arecord or ffmpeg" >&2; exit 1
fi

total_secs() {
    if command -v ffprobe >/dev/null; then
        find "$OUT" -name '*.wav' -exec ffprobe -v quiet -show_entries format=duration \
            -of csv=p=0 {} \; 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s}'
    else echo 0; fi
}

n=$(find "$OUT" -name 'clip_*.wav' 2>/dev/null | wc -l)
mapfile -t LINES < <(cat "$PROMPTS" 2>/dev/null || true)

echo "Recording into $OUT  (${SECS}s per take).  Ctrl-C to stop."
echo "Aim for 15+ minutes total. Keep the mic the same distance throughout."
echo

while true; do
    idx=$((n % ${#LINES[@]:-1}))
    echo "---------------------------------------------------------------"
    [ "${#LINES[@]}" -gt 0 ] && echo "READ:  ${LINES[$idx]}"
    printf "clip %03d — ENTER to record, or 'q' to quit: " "$n"
    read -r key </dev/tty
    [ "$key" = "q" ] && break

    f=$(printf "%s/clip_%03d.wav" "$OUT" "$n")
    echo "  ● recording ${SECS}s..."
    rec "$f"

    printf "  kept. [ENTER=next, r=redo] "
    read -r k </dev/tty
    if [ "$k" = "r" ]; then rm -f "$f"; continue; fi

    n=$((n + 1))
    t=$(total_secs)
    printf "  total so far: %dm %02ds\n\n" $((t / 60)) $((t % 60))
done

t=$(total_secs)
echo
printf "Done. %d clips, %dm %02ds total in %s\n" "$n" $((t / 60)) $((t % 60)) "$OUT"
[ "$t" -lt 600 ] && echo "NOTE: under 10 minutes — keep going, quality drops off sharply below that."
