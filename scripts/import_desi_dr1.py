import pyarrow as pa

from dask.distributed import Client
from hats_import import CollectionArguments, pipeline_with_client
from hats_import.catalog.file_readers import FitsReader
from upath import UPath


base_path = UPath("/ocean/projects/phy250012p/shared/3DTS/DESI/dr1")


class CustomFitsReader(FitsReader):
    """Reads fits and appends a new column with the DESI provenance"""

    def read(self, input_file, read_columns=None):
        for table in super().read(input_file, read_columns=read_columns):
            label = pa.DictionaryArray.from_arrays(
                pa.array([0] * table.num_rows, type=pa.int32()), pa.array(["DR1"])
            )
            yield table.append_column("DESI_RELEASE", label)


def import_desi_dr1():
    args = (
        CollectionArguments(
            output_artifact_name="desi_dr1_zcat",
            output_path=base_path,
            progress_bar=True,
            simple_progress_bar=True,
            resume=False,
        )
        .catalog(
            output_artifact_name="desi_dr1_zcat",
            input_file_list=[base_path / "iron_all_1.fits"],
            file_reader=CustomFitsReader(),
            ra_column="RA",
            dec_column="DEC",
            pixel_threshold=1_000_000,
            highest_healpix_order=8,
            skymap_alt_orders=[2, 4, 6],
            expected_total_rows=17_362_235,
        )
        .add_margin(margin_threshold=5.0, is_default=True)
        .add_index(indexing_column="TARGETID")
    )

    with Client(
        n_workers=4,
        threads_per_worker=1,
        local_directory=base_path / "dask_scratch",
        memory_limit=None,
    ) as client:
        pipeline_with_client(args, client)


if __name__ == "__main__":
    import_desi_dr1()
