#!/bin/bash -l
# One SCOTTY750 run: query the last three hours of ZTF, LSST and DECam alerts,
# screen them, and save the candidates to Fritz group 2064.
#
# A login shell (-l above) so ~/.bash_profile sets BOOM_USERNAME, BOOM_PASSWORD
# and FRITZ_TOKEN; the checkout as working directory so the relative paths in
# config.toml and scotty750.toml (output/, filters/, pipelines/) resolve there.
# Scheduled by scripts/scotty750.scrontab.
set -euo pipefail
REPO=/ocean/projects/phy250012p/xhall/GitHub/desi-alert-augmentation-pipeline
cd "$REPO"
# Any arguments are passed on, so the nightly sweep is
#   scripts/scotty750_hourly.sh -c scotty750_nightly.toml
exec /ocean/projects/phy250012p/xhall/envs/desi-aap/bin/desi-aap run -c config.toml -c scotty750.toml "$@"
