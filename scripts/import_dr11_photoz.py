"""Import the DR11 all-sky photo-z catalog as a HATS collection for the crossmatch stage.

The source is one FITS file (``dr11_photoz_allsky_v3.1.1.fits``, 42M rows, see the
README beside it). hats-import maps one task per input file, so a single 25 GB
file would be read by one worker, twice. The import is therefore two steps:

1. **shard** -- read the FITS table in row chunks (see FitsTable), keep the columns
   the pipeline needs, add the derived best-redshift columns below, and write one
   parquet shard per chunk;
2. **import** -- hats-import the shards in parallel, with a 5" margin.

The derived columns let the distance stage read this catalog like a DESI one (see
``[crossmatch.catalogs.dr11_photoz.redshift]`` in ``scotty750.toml``):

``Z``, ``Z_ERR``
    The DESI spectroscopic redshift where the catalog has a good one (``has_specz``
    and ``z_spec_flag == 0``, Loa = DR2 or Matterhorn = Y5, matched within 1"),
    otherwise the photo-z posterior median and its 68% half-width (``sigma``).
``Z_KIND``
    ``"spec"`` or ``"photo"``, saying which.
``Z_OK``
    The redshift is usable: a good spectrum, or a computed posterior (``fetch_ok``)
    with a finite median.
``SPECTYPE``
    The DESI spectral class of the kept Matterhorn spectrum, else the Loa one
    (GALAXY, QSO, STAR); empty where DESI never observed it. Lets the SCOTTY750
    screen veto QSO hosts the redshift catalogs missed.

Usage, on a compute node (it wants many cores and a few hundred GB of scratch)::

    python scripts/import_dr11_photoz.py shard
    python scripts/import_dr11_photoz.py import --workers 32
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

BASE = Path("/ocean/projects/phy250012p/shared/3DTS/DESI/dr11_photoz")
SOURCE = BASE / "dr11_photoz_allsky_v3.1.1.fits"
SHARDS = BASE / "shards_v3.1.1"
ARTIFACT = "dr11_photoz"

# Columns carried over unchanged.
KEEP = [
    "ls_id",
    "ra",
    "dec",
    "primary",
    "from_sga",
    "TYPE",
    "r_mag",
    "z_phot",
    "sigma",
    "z_phot_16",
    "z_phot_84",
    "fetch_ok",
    "z_spec",
    "has_specz",
    "z_spec_flag",
    "z_spec_source",
    "loa_targetid",
    "mh_targetid",
    "logm_saga",
    "novelty",
]
SPECTYPES = np.array(["GALAXY", "QSO", "STAR"])
CHUNK_ROWS = 1_000_000


class FitsTable:
    """Row chunks of a FITS binary table, read straight from the file.

    astropy's own reader cannot memory-map this file on /ocean and falls back to
    reading it whole, so the rows are read here with ``numpy.fromfile`` at the
    offset the header gives, and each column decoded from its TFORM/TZERO.
    """

    def __init__(self, path: Path, hdu: str | int = "PHOTOZ"):
        self.path = path
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            header = hdul[hdu].header
            self.offset = hdul.fileinfo(hdul.index_of(hdu))["datLoc"]
            # The ColDefs dtype is native-endian; FITS is big-endian on disk.
            self.dtype = hdul[hdu].columns.dtype.newbyteorder(">")
        self.n_rows = header["NAXIS2"]
        self.formats = {}
        for i in range(1, header["TFIELDS"] + 1):
            self.formats[header[f"TTYPE{i}"]] = (header[f"TFORM{i}"].strip()[-1], header.get(f"TZERO{i}"))
        if self.dtype.itemsize != header["NAXIS1"]:
            raise ValueError(f"row size {self.dtype.itemsize} does not match NAXIS1 {header['NAXIS1']}")

    def read(self, start: int, count: int) -> np.ndarray:
        """Raw records ``start`` to ``start + count``."""
        count = min(count, self.n_rows - start)
        return np.fromfile(
            self.path, dtype=self.dtype, count=count, offset=self.offset + start * self.dtype.itemsize
        )

    def column(self, rows: np.ndarray, name: str) -> np.ndarray:
        """One column decoded: logicals to bool, offset bytes to ints, strings stripped."""
        code, zero = self.formats[name]
        values = rows[name]
        if code == "L":
            return values == ord("T")
        if code == "A":
            return np.char.strip(np.char.decode(values, "ascii"))
        values = values.astype(values.dtype.newbyteorder("="))
        if zero is not None:
            values = values.astype(np.int64) + int(zero)
        return values


def _spectype(code: np.ndarray) -> np.ndarray:
    """DESI spectype codes (0 GALAXY, 1 QSO, 2 STAR, -1 none) as names."""
    code = np.asarray(code).astype(int)
    out = np.full(code.shape, "", dtype=object)
    known = (code >= 0) & (code < len(SPECTYPES))
    out[known] = SPECTYPES[code[known]]
    return out


def derive(table: FitsTable, rows: np.ndarray) -> pa.Table:
    """The kept columns of one chunk, plus Z, Z_ERR, Z_KIND, Z_OK and SPECTYPE."""
    columns = {name: pa.array(table.column(rows, name)) for name in KEEP}
    z_spec = table.column(rows, "z_spec").astype(float)
    z_phot = table.column(rows, "z_phot").astype(float)
    sigma = table.column(rows, "sigma").astype(float)
    has_spec = (
        table.column(rows, "has_specz") & (table.column(rows, "z_spec_flag") == 0) & np.isfinite(z_spec)
    )
    photo_ok = table.column(rows, "fetch_ok") & np.isfinite(z_phot)
    columns["Z"] = pa.array(np.where(has_spec, z_spec, z_phot))
    columns["Z_ERR"] = pa.array(np.where(has_spec, np.nan, sigma))
    columns["Z_KIND"] = pa.array(np.where(has_spec, "spec", "photo"))
    columns["Z_OK"] = pa.array(has_spec | photo_ok)
    mh = _spectype(table.column(rows, "mh_spectype"))
    loa = _spectype(table.column(rows, "loa_spectype"))
    columns["SPECTYPE"] = pa.array(np.where(mh != "", mh, loa).astype(str))
    return pa.table(columns)


def shard() -> None:
    """Write the FITS table as parquet shards of CHUNK_ROWS rows each."""
    SHARDS.mkdir(parents=True, exist_ok=True)
    table = FitsTable(SOURCE)
    for i, start in enumerate(range(0, table.n_rows, CHUNK_ROWS)):
        path = SHARDS / f"part_{i:04d}.parquet"
        if path.exists():
            continue
        chunk = derive(table, table.read(start, CHUNK_ROWS))
        tmp = path.with_suffix(".tmp")
        pq.write_table(chunk, tmp)
        tmp.rename(path)
        print(f"wrote {path.name}: rows {start}-{start + len(chunk) - 1} of {table.n_rows}", flush=True)


def import_collection(workers: int) -> None:
    """hats-import the shards into a collection with a 5" margin."""
    from dask.distributed import Client
    from hats_import import CollectionArguments, pipeline_with_client

    args = (
        CollectionArguments(
            output_artifact_name=ARTIFACT,
            output_path=BASE,
            progress_bar=True,
            simple_progress_bar=True,
            resume=True,
        )
        .catalog(
            output_artifact_name=ARTIFACT,
            input_file_list=sorted(SHARDS.glob("part_*.parquet")),
            file_reader="parquet",
            ra_column="ra",
            dec_column="dec",
            pixel_threshold=1_000_000,
            highest_healpix_order=8,
            skymap_alt_orders=[2, 4, 6],
            npix_suffix="/",
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        # No ls_id index: the pipeline only cross-matches by position, and
        # hats-import's index step pins an old dask (<2025.4 on Python 3.11).
    )
    with Client(
        n_workers=workers, threads_per_worker=1, local_directory=BASE / "dask_scratch", memory_limit=None
    ) as client:
        pipeline_with_client(args, client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["shard", "import"])
    parser.add_argument("--workers", type=int, default=32)
    options = parser.parse_args()
    if options.step == "shard":
        shard()
    else:
        import_collection(options.workers)
