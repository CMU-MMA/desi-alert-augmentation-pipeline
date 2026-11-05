#!/bin/bash -l
# Run SCOTTY750 at five past every hour, forever, with the nightly 7-day sweep
# in place of the hourly run once a day. Meant for a tmux session on a machine
# that stays up:
#
#   tmux new -d -s scotty750 scripts/scotty750_loop.sh
#   tmux attach -t scotty750        # watch it; Ctrl-b d to leave it running
#   tmux kill-session -t scotty750  # stop it
#
# The sweep runs at SWEEP_HOUR_UTC (default 18, after the ZTF, LSST and DECam
# nights have all ended) and covers the last week, so it also covers the three
# hours the hourly run would have.
#
# Each run's pipeline log is under output/scotty750/logs/; this loop's own log,
# output/scotty750/loop.log, says when each run started, which kind it was,
# and how it ended. A failed run is logged and the loop carries on. The sleep
# is measured from the end of a run, so two runs never overlap.
REPO=/ocean/projects/phy250012p/xhall/GitHub/desi-alert-augmentation-pipeline
LOG="$REPO/output/scotty750/loop.log"
SWEEP_HOUR_UTC=${SWEEP_HOUR_UTC:-18}
mkdir -p "$(dirname "$LOG")"
while true; do
    if [ "$((10#$(date -u +%H)))" -eq "$SWEEP_HOUR_UTC" ]; then
        kind="nightly sweep"
        args=(-c scotty750_nightly.toml)
    else
        kind="run"
        args=()
    fi
    echo "$(date -u +%FT%TZ) $kind starting" >> "$LOG"
    if "$REPO/scripts/scotty750_hourly.sh" "${args[@]}" >> "$LOG" 2>&1; then
        echo "$(date -u +%FT%TZ) $kind finished" >> "$LOG"
    else
        echo "$(date -u +%FT%TZ) $kind FAILED (exit $?)" >> "$LOG"
    fi
    # Sleep until five past the next hour.
    now=$(date +%s)
    next=$(( (now / 3600 + 1) * 3600 + 300 ))
    sleep $(( next - now ))
done
