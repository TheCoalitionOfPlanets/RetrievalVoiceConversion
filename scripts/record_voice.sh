#!/usr/bin/env bash
# Record training audio for an RVC target voice.
#
#   ./scripts/record_voice.sh [outdir]
#
# Shows a paragraph, ENTER starts recording, ENTER stops it. Then keep or redo.
# Mono 48 kHz WAV (the loader resamples to 32 kHz). Resumable: re-running
# continues from the next free clip number and keeps counting toward the goal.

set -uo pipefail

OUT="${1:-data/raw/myvoice}"
PROMPTS="${PROMPTS:-scripts/prompts.txt}"
GOAL_MIN="${GOAL_MIN:-15}"

mkdir -p "$OUT"

[ -f "$PROMPTS" ] || { echo "prompts file not found: $PROMPTS" >&2; exit 1; }

# --- recording backend -------------------------------------------------------
REC_PID=""
if command -v arecord >/dev/null 2>&1; then
    start_rec() { arecord -q -f S16_LE -c1 -r48000 "$1" & REC_PID=$!; }
elif command -v ffmpeg >/dev/null 2>&1; then
    start_rec() { ffmpeg -hide_banner -loglevel error -f pulse -i default \
                         -ac 1 -ar 48000 -y "$1" </dev/null & REC_PID=$!; }
else
    echo "need arecord (alsa-utils) or ffmpeg" >&2; exit 1
fi

stop_rec() {
    [ -n "$REC_PID" ] || return 0
    kill -INT "$REC_PID" 2>/dev/null
    wait "$REC_PID" 2>/dev/null
    REC_PID=""
}
trap 'stop_rec; echo; echo "stopped."; exit 0' INT TERM

dur_of() {
    command -v ffprobe >/dev/null 2>&1 || { echo 0; return; }
    ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null \
        | awk '{printf "%.0f", $1+0}'
}

total_secs() {
    local t=0 f
    for f in "$OUT"/clip_*.wav; do
        [ -e "$f" ] || continue
        t=$((t + $(dur_of "$f")))
    done
    echo "$t"
}

# --- load paragraphs (blank-line separated) ----------------------------------
mapfile -t PARAS < <(awk 'BEGIN{RS=""} {gsub(/\n/," "); gsub(/  +/," "); print}' "$PROMPTS")
[ "${#PARAS[@]}" -gt 0 ] || { echo "no paragraphs in $PROMPTS" >&2; exit 1; }

n=0
while [ -e "$(printf '%s/clip_%03d.wav' "$OUT" "$n")" ]; do n=$((n+1)); done

goal=$((GOAL_MIN * 60))
t=$(total_secs)

echo "================================================================"
echo " Recording into : $OUT"
echo " Paragraphs     : ${#PARAS[@]} (cycles; repeats are fine, timbre is"
echo "                  what matters, not unique text)"
echo " Goal           : ${GOAL_MIN} min   |   already have: $((t/60))m $((t%60))s"
echo "================================================================"
echo " Same mic, same distance, same room. Silent gaps. Speak normally."
echo "================================================================"
echo

while true; do
    t=$(total_secs)
    pct=0; [ "$goal" -gt 0 ] && pct=$((t * 100 / goal)); [ "$pct" -gt 100 ] && pct=100
    bar=$(printf '%*s' $((pct / 4)) '' | tr ' ' '#')
    printf '[%-25s] %d%%  (%dm %02ds / %dm)\n' "$bar" "$pct" $((t/60)) $((t%60)) "$GOAL_MIN"
    echo "----------------------------------------------------------------"
    printf 'clip %03d — READ THIS:\n\n' "$n"
    printf '%s\n' "${PARAS[$((n % ${#PARAS[@]}))]}" | fold -s -w 64 | sed 's/^/    /'
    echo
    printf '  ENTER = start recording   |   q = quit : '
    read -r key </dev/tty
    [ "$key" = "q" ] && break

    f=$(printf '%s/clip_%03d.wav' "$OUT" "$n")
    start_rec "$f"
    printf '  \033[31m● RECORDING\033[0m — read it, then press ENTER to stop: '
    read -r _ </dev/tty
    stop_rec

    d=$(dur_of "$f")
    if [ "${d:-0}" -lt 2 ]; then
        echo "  !! only ${d}s — discarded (too short)."
        rm -f "$f"; echo; continue
    fi

    printf '  got %ss.  ENTER = keep & next   |   r = redo : ' "$d"
    read -r k </dev/tty
    if [ "$k" = "r" ]; then rm -f "$f"; echo "  redoing."; echo; continue; fi

    n=$((n + 1))
    echo
done

stop_rec
t=$(total_secs)
echo
echo "================================================================"
printf ' %d clips, %dm %02ds total in %s\n' "$n" $((t/60)) $((t%60)) "$OUT"
if [ "$t" -lt 600 ]; then
    echo " UNDER 10 MIN — quality drops off sharply below this. Keep going."
elif [ "$t" -lt "$goal" ]; then
    echo " Usable, but more is better. Re-run to continue where you left off."
else
    echo " Good amount. Next: uv run python scripts/check_voice.py $OUT"
fi
echo "================================================================"
