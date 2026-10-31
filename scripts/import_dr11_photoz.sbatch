#!/bin/bash
#SBATCH --job-name=dr11_photoz_import
#SBATCH --partition=RM-shared
#SBATCH --account=phy250012p
#SBATCH --ntasks-per-node=32
#SBATCH --time=05:00:00
#SBATCH --output=/ocean/projects/phy250012p/shared/3DTS/DESI/dr11_photoz/import_%j.log
# Shard the DR11 photo-z FITS file and import it as a HATS collection.
#   sbatch scripts/import_dr11_photoz.sbatch
set -euo pipefail
cd /ocean/projects/phy250012p/xhall/GitHub/desi-alert-augmentation-pipeline
PY=/ocean/projects/phy250012p/xhall/envs/desi-aap/bin/python
[ -n "${SKIP_SHARD:-}" ] || $PY scripts/import_dr11_photoz.py shard
$PY scripts/import_dr11_photoz.py import --workers 28
