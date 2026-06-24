import pyarrow as pa

from astropy.io import fits
from astropy.table import Table, join
from dask.distributed import Client
from hats_import import CollectionArguments, pipeline_with_client
from hats_import.catalog.file_readers import InputReader
from hats_import.catalog.file_readers.fits import _astropy_to_pyarrow_table

from upath import UPath


base_path = UPath("/ocean/projects/phy250012p/shared/3DTS/DESI/dr2")


class CustomFitsReader(InputReader):
    def __init__(self, sorted_columns: list[str] | None = None):
        self.sorted_columns = sorted_columns

    def read(self, input_file, read_columns=None):
        input_file = self.regular_file_exists(input_file)

        with input_file.open("rb") as file_handle, fits.open(file_handle) as hdul:
            meta_hdu = Table.read(hdul["METADATA"])
            specphot_hdu = Table.read(hdul["SPECPHOT"])
            fastspec_hdu = Table.read(hdul["FASTSPEC"])

            # Strip the header-checksum keys from each HDU before joining
            for t in (meta_hdu, specphot_hdu, fastspec_hdu):
                for k in ("DATASUM", "CHECKSUM", "EXTNAME"):
                    t.meta.pop(k, None)

            # Join data from each HDU
            merge_keys = ["TARGETID", "SURVEY", "PROGRAM", "HEALPIX"]
            table = join(meta_hdu, specphot_hdu, keys=merge_keys, join_type="inner")
            table = join(table, fastspec_hdu, keys=merge_keys, join_type="inner")

            # Append a new column with the DESI provenance
            table = _astropy_to_pyarrow_table(table, flatten_tensors=False)
            label = pa.DictionaryArray.from_arrays(
                pa.array([0] * table.num_rows, type=pa.int32()), pa.array(["DR2"])
            )
            table = table.append_column("DESI_RELEASE", label)

            # Order columns deterministically
            yield table.select(self.sorted_columns or table.column_names)


def import_desi_dr2():
    input_file_list = list((base_path / "catalogs").glob("*.fits"))
    print(f"Found {len(input_file_list)} files for import")

    # Read an input file to extract all column names
    table = next(CustomFitsReader().read(input_file_list[28]))

    args = (
        CollectionArguments(
            output_artifact_name="desi_dr2_zcat",
            output_path=base_path,
            progress_bar=True,
            simple_progress_bar=True,
            resume=True,
        )
        .catalog(
            output_artifact_name="desi_dr2_zcat",
            input_file_list=input_file_list,
            file_reader=CustomFitsReader(table.column_names),
            ra_column="RA",
            dec_column="DEC",
            pixel_threshold=400_000,
            highest_healpix_order=8,
            skymap_alt_orders=[2, 4, 6],
            npix_suffix="/",
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        .add_index(indexing_column="TARGETID")
    )

    with Client(
        n_workers=32,
        threads_per_worker=1,
        local_directory=base_path / "dask_scratch",
        memory_limit=None,
    ) as client:
        pipeline_with_client(args, client)


if __name__ == "__main__":
    import_desi_dr2()
