#!/bin/bash
# Runs the 2B inside the merged 35B file on the 7120A, 60 cores x 2 threads: reserves 2 MiB pages on the card, starts the host's side, then the card's program,
# prints what the run says, stops the card, puts the page reservation back.
#
#   scripts/chip_card_run.sh [-m MEMO] [-w WIDTH] [-p PAR_MIN] [-n MAX_TOKENS] [-g GEN] [-l LOGDIR] [-r REF_HASH_FILE] "PROMPT"
#
# REF_HASH_FILE: q2b64_check --nofix --hashlog for the same prompt and file: the hash is compared live and a difference stops the run. LOGDIR/ckpt.bin, if there, is resumed.
set -uo pipefail
cd "$(dirname "$0")/.."
CORES=60; TPC=2; MEMO=1; WIDTH=120; PARMIN=64; MAX=1152921504606846976; GEN=64; LOG=logs/card; REF=
# 3.56 GB: the image with its memories fits with room to spare (the host says how much it took), far inside the 12 GB budget for the card.
PAGES=1700
while getopts "m:w:p:n:g:l:P:r:" o; do
    case $o in
        m) MEMO=$OPTARG ;;
        w) WIDTH=$OPTARG ;;
        p) PARMIN=$OPTARG ;;
        n) MAX=$OPTARG ;;
        g) GEN=$OPTARG ;;
        l) LOG=$OPTARG ;;
        P) PAGES=$OPTARG ;;
        r) REF=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
PROMPT=$1
if [ "$PAGES" -gt 5722 ]; then
    echo "card: refused: $PAGES 2 MiB pages are over the 12 GB maximum (5,722); nothing was started" >&2
    exit 3
fi
mkdir -p "$LOG"
WAS=$(phi run cat /proc/sys/vm/nr_hugepages)
# four more than the arena: the DMA ring takes one
phi run sh -c "echo $((PAGES + 4)) > /proc/sys/vm/nr_hugepages; grep -E 'HugePages_(Total|Free)|MemFree' /proc/meminfo" | tr '\n' ' ' | tr -s ' ' | cut -c1-110; echo
phi put out/card/chip_card /tmp/chip_card >/dev/null
phi run chmod +x /tmp/chip_card
out/host/chip_host out/merged/q35-with-2b.gguf "$PROMPT" --gen "$GEN" --hashlog "$LOG/hash.txt" ${REF:+--ref "$REF"} --reply "$LOG/reply.txt" --started "$LOG/started.txt" --ckpt "$LOG/ckpt.bin" &
HOST=$!
sleep 1
phi run sh -c "rm -f /tmp/chip.out /tmp/chip.rc; (/tmp/chip_card $CORES $TPC $PAGES $WIDTH $PARMIN $MEMO $MAX > /tmp/chip.out 2>&1; echo \$? > /tmp/chip.rc) </dev/null >/dev/null 2>&1 &" >/dev/null
wait "$HOST"
RC=$?
sleep 1
phi run cat /tmp/chip.out 2>/dev/null | cut -c1-140
phi run sh -c "cat /tmp/chip.rc 2>/dev/null | sed 's/^/card exit /'; killall chip_card 2>/dev/null; sleep 1; echo $WAS > /proc/sys/vm/nr_hugepages; echo card huge pages back to $WAS"
exit "$RC"
