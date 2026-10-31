#!/bin/bash -l
# Run SCOTTY750 at five past every hour, forever. Meant for a tmux session on a
# machine that stays up:
#
#   tmux new -d -s scotty750 scripts/scotty750_loop.sh
#   tmux attach -t scotty750        # watch it; Ctrl-b d to leave it running
#   tmux kill-session -t scotty750  # stop it
#
# Each run's pipeline log is under output/scotty750/logs/; this loop's own log,
# output/scotty750/loop.log, says when each run started and how it ended. A
# failed run is logged and the loop carries on to the next hour. The sleep is
# measured from the end of a run, so two runs never overlap.
REPO=/ocean/projects/phy250012p/xhall/GitHub/desi-alert-augmentation-pipeline
LOG="$REPO/output/scotty750/loop.log"
mkdir -p "$(dirname "$LOG")"
while true; do
    echo "$(date -u +%FT%TZ) run starting" >> "$LOG"
    if "$REPO/scripts/scotty750_hourly.sh" >> "$LOG" 2>&1; then
        echo "$(date -u +%FT%TZ) run finished" >> "$LOG"
    else
        echo "$(date -u +%FT%TZ) run FAILED (exit $?)" >> "$LOG"
    fi
    # Sleep until five past the next hour.
    now=$(date +%s)
    next=$(( (now / 3600 + 1) * 3600 + 300 ))
    sleep $(( next - now ))
done
