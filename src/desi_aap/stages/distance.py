"""Put every cross-matched alert at a distance, using its nearest DESI host.

An alert carries no redshift of its own. What it carries, after the crossmatch
stage, is one or more DESI sources that sit close to it on the sky, and those do
have redshifts. This stage picks one of them per alert (:func:`nearest_hosts`)
and turns its redshift into a luminosity distance under each cosmology in
:data:`desi_aap.cosmology.COSMOLOGIES` (:func:`attach_distances`).

It exists as its own stage because every filter downstream is a cut on
distance in some form -- a GW credible *volume*, an absolute magnitude, a
luminosity-distance limit -- so the host has to be chosen once, the same way,
before any of them run. Choosing it inside one filter would mean the others
either repeated the work or disagreed with it.

An alert with no usable host leaves here: without a redshift there is no
distance, and without a distance none of the filters downstream can say
anything about it. The count of alerts dropped that way is in the stage's
summary, so a run that loses most of its alerts to missing hosts says so.

The output keeps the shape the crossmatch stage established -- one row per
alert, each catalog's matches in its own nested column -- with the chosen host
and its distances added as flat columns::

    objectId  host_catalog  host_redshift  host_sep_arcsec  dist_mpc_SHOES  dist_mpc_Planck18  desi_dr1
    LSST001   desi_dr1              0.030            0.412           130.4              135.9  [{...}]
    LSST002   desi_dr2              0.021            1.883            91.2               95.0  [{...}]

``desi_dr1`` is the crossmatch stage's nested column, untouched.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy import units as u

from desi_aap.boom import ALERT_MAG_COLUMN
from desi_aap.config import PipelineConfig, RedshiftFieldsConfig
from desi_aap.cosmology import COSMOLOGIES
from desi_aap.stages.base import StageInputs, StageResult, input_result, write_frame
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.crossmatch import catalog_specs
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

STAGE = "distance"

# Stages whose output this one consumes. See crossmatch.REQUIRES.
REQUIRES: tuple[str, ...] = (CROSSMATCH_STAGE,)

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "distances"

# Flat columns this stage adds describing the chosen host. The distance columns
# are named per cosmology and built by dist_column below, since which of them
# exist follows COSMOLOGIES rather than being fixed here.
HOST_COLUMNS = (
    "host_catalog",
    "host_redshift",
    "host_redshift_err",
    "host_redshift_kind",
    "host_sep_arcsec",
    "host_spectype",
)


def dist_column(label: str) -> str:
    """Name of the luminosity-distance column for one cosmology.

    Parameters
    ----------
    label : str
        A key of :data:`desi_aap.cosmology.COSMOLOGIES`, such as ``"SHOES"``.

    Returns
    -------
    str
        The column name, such as ``"dist_mpc_SHOES"``.
    """
    return f"dist_mpc_{label}"


# Every distance column this stage writes, one per configured cosmology.
DIST_COLUMNS = tuple(dist_column(label) for label in COSMOLOGIES)


def abs_mag_column(label: str) -> str:
    """Name of the absolute-magnitude column for one cosmology.

    Parameters
    ----------
    label : str
        A key of :data:`desi_aap.cosmology.COSMOLOGIES`, such as ``"SHOES"``.

    Returns
    -------
    str
        The column name, such as ``"abs_mag_SHOES"``.
    """
    return f"abs_mag_{label}"


# Every absolute-magnitude column this stage writes, one per cosmology, from
# the alert's apparent magnitude at that cosmology's luminosity distance. No
# K-correction and no extinction: this is the screening number the filters cut
# on, not a measurement to publish. Band-agnostic for the same reason -- the
# band is carried beside it for whoever wants to be more careful.
ABS_MAG_COLUMNS = tuple(abs_mag_column(label) for label in COSMOLOGIES)


def _host_fields(matches: npd.NestedFrame, name: str, fields: Sequence[str]) -> pd.DataFrame:
    """Flatten one catalog's nested column, checking it carries the fields wanted.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    name : str
        Nested column to read hosts from, i.e. the configured catalog name.
    fields : sequence of str
        Fields the nested column must carry: the redshift, the flag saying
        whether it is trustworthy, and the angular separation between alert and
        host, plus any per-row kind column the catalog is configured with.

    Returns
    -------
    pandas.DataFrame
        Flattened nested column, indexed by alert row. Empty DataFrame if the
        nested column is empty.

    Raises
    ------
    ValueError
        If the nested column is missing any of ``fields``.
    """
    flat = matches[name].nest.to_flat()
    missing = [field for field in fields if field not in flat.columns]
    if missing:
        raise ValueError(
            f"Nested column {name!r} has no {', '.join(repr(f) for f in missing)} field, so "
            f"stage {STAGE!r} cannot read a host redshift from it. Every catalog in "
            "[crossmatch.catalogs.*] is read for one: name its columns in that catalog's "
            "[crossmatch.catalogs.<name>.redshift] table if they differ from DESI's."
        )
    return flat


def _optional(flat: pd.DataFrame, column: str | None) -> pd.Series:
    """A column of ``flat`` if configured and present, else all-missing."""
    if column is not None and column in flat.columns:
        return flat[column]
    return pd.Series(pd.NA, index=flat.index, dtype="object")


def nearest_hosts(
    matches: npd.NestedFrame,
    catalog_names: Sequence[str],
    *,
    min_redshift: float,
    redshift_field: str = "Z",
    warning_field: str = "ZWARN",
    separation_field: str = "_dist_arcsec",
    ok_warning: int = 0,
    fields: Mapping[str, RedshiftFieldsConfig] | None = None,
    spec_preference_arcsec: float = 0.0,
) -> pd.DataFrame:
    """Pick the host each alert takes its redshift from.

    An alert can carry several host candidates (one per neighbour, per catalog)
    and a distance needs exactly one. The candidates from every catalog are
    pooled and the closest one on the sky wins, so which catalog an alert's
    redshift comes from is decided by the association rather than by the order in
    which the catalogs are configured.

    The one exception is a spectroscopic redshift against a photometric one. The
    same galaxy sits in a spectroscopic and a photometric catalog at positions
    that differ by astrometry alone, and the spectrum is the better redshift, so
    a spectroscopic host within ``spec_preference_arcsec`` of the nearest host's
    separation wins over it. With spectroscopic catalogs alone this changes
    nothing.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    catalog_names : sequence of str
        Which catalogs to read hosts from, named as they are in
        [crossmatch.catalogs.*] -- each is a nested column on ``matches``. A
        catalog with no column on the frame is skipped, since a crossmatch that
        produced no column for it produced no hosts either. A catalog whose
        column is present but lacks a required field raises instead: that is a
        schema mismatch rather than an absence.
    min_redshift : float
        Smallest redshift treated as a real measurement, inclusive. Hosts below
        it are discarded rather than clamped, for the reasons set out in
        tns_catalog.clean_tns_catalog: a non-positive redshift yields no usable
        luminosity distance, and small negative ones make ``SkyCoord`` raise
        further down.
    redshift_field, warning_field, ok_warning : str, str, int
        Where a catalog with no entry in ``fields`` keeps its redshift, and the
        flag column and value meaning that redshift is trustworthy. The DESI
        redshift catalog's own names.
    separation_field : str
        The angular separation between alert and host, which is what LSDB's
        crossmatch_nested records for every match it makes.
    fields : mapping of str to RedshiftFieldsConfig, optional
        Per-catalog column names, overriding the three above and adding the
        redshift error, its kind (spec or photo), and the spectral class.
    spec_preference_arcsec : float
        How much farther than the nearest host a spectroscopic one may sit and
        still win. See above.

    Returns
    -------
    pandas.DataFrame
        One row per alert that has a usable host, indexed by that alert's row in
        ``matches``, with the :data:`HOST_COLUMNS`:

        host_catalog
            The nested column the host came from.
        host_redshift
            Its redshift.
        host_redshift_err
            Its uncertainty, missing where the catalog has none.
        host_redshift_kind
            "spec" or "photo".
        host_sep_arcsec
            Its separation from the alert, which is what it won on.
        host_spectype
            Its spectral class (GALAXY, QSO, STAR), missing where the catalog
            has none.

        An alert appears here only if at least one of its hosts passed the
        cuts. One whose hosts all failed is left out entirely rather than kept
        with a missing redshift, which is what lets the caller read a missing
        row as "this alert has no usable host". If no alert qualifies the result
        is empty but still carries the columns above.

    Raises
    ------
    ValueError
        If a named nested column is missing a required field.
    """
    fields = fields or {}
    candidates = []
    for name in catalog_names:
        if name not in matches.columns or matches.empty:
            continue
        spec = fields.get(name) or RedshiftFieldsConfig(
            column=redshift_field, ok_column=warning_field, ok_value=ok_warning
        )
        required = [spec.column, spec.ok_column, separation_field]
        if spec.kind_column is not None:
            required.append(spec.kind_column)
        flat = _host_fields(matches, name, required)
        if flat.empty:  # No candidates from this catalog, so it cannot contribute to the nearest host.
            continue
        kind = flat[spec.kind_column].astype("object") if spec.kind_column else spec.kind
        candidates.append(
            pd.DataFrame(
                {
                    "host_catalog": name,
                    "host_redshift": pd.to_numeric(flat[spec.column], errors="coerce"),
                    "host_redshift_err": pd.to_numeric(_optional(flat, spec.error_column), errors="coerce"),
                    "host_redshift_kind": kind,
                    "host_sep_arcsec": pd.to_numeric(flat[separation_field], errors="coerce"),
                    "host_spectype": _optional(flat, spec.spectype_column).astype("object"),
                    "_ok": flat[spec.ok_column].eq(spec.ok_value),
                },
                index=flat.index,
            )
        )

    if not candidates:
        return pd.DataFrame(columns=list(HOST_COLUMNS))

    hosts = pd.concat(candidates)
    # NaN fails all three comparisons, so an unparseable redshift, an absent
    # flag and a missing separation all leave by this line.
    hosts = hosts[
        hosts["_ok"].fillna(False).astype(bool)
        & (hosts["host_redshift"] >= min_redshift)
        & hosts["host_sep_arcsec"].notna()
        & hosts["host_redshift_kind"].isin(["spec", "photo"])
    ]
    if hosts.empty:
        return pd.DataFrame(columns=list(HOST_COLUMNS))

    # Rank on an effective separation: a spectroscopic host is treated as up to
    # spec_preference_arcsec closer than it is, so it beats a photometric host of
    # the same galaxy, while any host still beats a farther one of its own kind.
    # Sorted on the catalog name as well so that two hosts at the same separation
    # -- the same object present in two releases -- resolve the same way on every
    # run rather than on whichever release happened to be concatenated first.
    is_spec = hosts["host_redshift_kind"].eq("spec")
    hosts = hosts.assign(
        _rank=hosts["host_sep_arcsec"] - is_spec * spec_preference_arcsec,
        _spec_first=~is_spec,
    )
    hosts = hosts.sort_values(["_rank", "_spec_first", "host_sep_arcsec", "host_catalog"], kind="stable")
    hosts = hosts[~hosts.index.duplicated(keep="first")]
    return hosts[list(HOST_COLUMNS)].sort_index()


def attach_distances(matches: npd.NestedFrame, hosts: pd.DataFrame) -> npd.NestedFrame:
    """Put the chosen host, its distances, and the implied brightness on each alert row.

    Every cosmology in :data:`desi_aap.cosmology.COSMOLOGIES` gets its own
    column rather than one being chosen here, because the choice belongs to
    whoever is cutting on the number: the GW match reports its result under each
    in turn, while a magnitude or distance limit names the one it means. Two
    columns of the same redshift are cheap; a distance silently computed under
    the wrong cosmology is not.

    The absolute magnitude is the distance modulus applied to the alert's
    apparent PSF magnitude, ``M = m - 5 (log10(d_Mpc) + 5)`` -- no K-correction
    and no extinction, so it is a screening number rather than a measurement.
    It lives here rather than in any filter for the same reason the distance
    does: every filter that cuts on brightness must mean the same brightness.
    An alert with no usable apparent magnitude gets ``NaN``, which fails every
    cut downstream rather than raising; a frame with no magnitude *column* gets
    no absolute-magnitude columns at all, so a brightness filter fails loudly
    instead of running quiet forever.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    hosts : pandas.DataFrame
        Chosen hosts from :func:`nearest_hosts`, indexed by alert row.

    Returns
    -------
    nested_pandas.NestedFrame
        The alerts that had a usable host, in their original order, with
        :data:`HOST_COLUMNS`, :data:`DIST_COLUMNS` and :data:`ABS_MAG_COLUMNS`
        added. The join is an inner one, so an alert absent from ``hosts`` is
        dropped: it has no redshift, so no distance, and nothing downstream
        could place it. Empty NestedFrame carrying those columns if no alert
        had a host.
    """
    joined = npd.NestedFrame(matches.join(hosts, how="inner"))
    # A frame with no magnitude column still gets its distances, but no
    # absolute magnitudes AT ALL -- absent columns rather than all-NaN ones.
    # All-NaN columns would let a brightness filter run and find nothing,
    # forever, looking exactly like a quiet sky; absent columns make the same
    # filter fail loudly, naming the column, while filters that never cut on
    # brightness are unaffected.
    has_apparent = ALERT_MAG_COLUMN in joined.columns
    if not has_apparent:
        logger.warning(
            "Alert frame has no %r column, so no absolute magnitudes are computed. "
            "Any filter cutting on abs_mag_* will fail; check the projection in "
            "default_pipeline.json.",
            ALERT_MAG_COLUMN,
        )
    else:
        apparent = pd.to_numeric(joined[ALERT_MAG_COLUMN], errors="coerce").astype(float).to_numpy()

    for label, cosmo in COSMOLOGIES.items():
        if joined.empty:
            # astropy's luminosity_distance goes through np.vectorize, which
            # rejects a size-0 input rather than returning an empty result.
            joined[dist_column(label)] = np.array([], dtype=float)
            if has_apparent:
                joined[abs_mag_column(label)] = np.array([], dtype=float)
            continue
        dist_mpc = cosmo.luminosity_distance(joined["host_redshift"].to_numpy()).to_value(u.Mpc)
        joined[dist_column(label)] = dist_mpc
        if has_apparent:
            # The distance modulus, with the distance converted from Mpc to
            # the 10 pc the magnitude scale is defined against.
            joined[abs_mag_column(label)] = apparent - 5 * (np.log10(dist_mpc) + 5)

    return joined


def run_distance(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run the stage: choose each alert's host, and put the alert at its distance.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The pipeline configuration, normally from
        :func:`desi_aap.config.load_config`.
    dry_run : bool
        Do the work but write nothing. The distances are still computed, so a
        dry run reports the same summary a real one would.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The alerts come from
        ``crossmatch``.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The alerts that have a distance, and where they were written. ``frame``
        is ``None`` when there were no matched alerts to place.

    Raises
    ------
    KeyError
        If ``crossmatch`` has not run.
    ValueError
        If no catalogs are configured to take host redshifts from, or if one of
        those catalogs carries no redshift.
    """
    settings = cfg.distance
    catalog_names = [spec.name for spec in catalog_specs(cfg)]
    stage_dir = cfg.run.stage_dir(STAGE)

    stamp = stamp or run_stamp()
    (upstream,) = REQUIRES
    matches = input_result(inputs, upstream).frame

    if matches is None or matches.empty:
        logger.info("No matched alerts to place at a distance; writing nothing.")
        return StageResult(stage=STAGE, frame=None, stamp=stamp, summary={"n_alerts": 0})

    hosts = nearest_hosts(
        matches,
        catalog_names,
        min_redshift=settings.min_redshift,
        fields={name: entry.redshift for name, entry in cfg.crossmatch.catalogs.items()},
        spec_preference_arcsec=settings.spec_preference_arcsec,
    )
    frame = attach_distances(matches, hosts)

    summary: dict[str, Any] = {
        "n_alerts": len(matches),
        "n_alerts_with_host": len(frame),
        "n_alerts_without_host": len(matches) - len(frame),
    }
    if not frame.empty:
        summary["hosts_by_catalog"] = frame["host_catalog"].value_counts().to_dict()
    logger.info("Distance summary: %s", summary)
    logger.info(
        "%d of %d matched alerts have a usable host redshift.",
        len(frame),
        len(matches),
    )

    output_path: Path | None = None
    if dry_run:
        logger.info("Dry run: not writing the distances.")
    elif frame.empty:
        logger.info("No matched alert had a usable host redshift; writing nothing.")
    else:
        output_path = write_frame(frame, stage_dir / f"{OUTPUT_PREFIX}_{stamp}.parquet")
        logger.info("Wrote %d alerts with distances to %s.", len(frame), output_path)

    return StageResult(stage=STAGE, frame=frame, output_path=output_path, stamp=stamp, summary=summary)
