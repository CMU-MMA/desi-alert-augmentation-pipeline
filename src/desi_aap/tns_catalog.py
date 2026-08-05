"""Download and filter the TNS public catalog for stripped-envelope supernovae (SESN).

The Transient Name Server publishes its whole public object catalog as a zipped CSV that is
regenerated daily after UT midnight. Downloading it requires a registered TNS bot.
"""

import warnings

import numpy as np
import pandas as pd
import requests
from astropy import units as u

from desi_aap.cosmology import COSMOLOGIES

# TNS settings copied from the existing workflow.
TNS_API_KEY = "174094777967c4c1438cdfd6.00935564"
TNS_BOT_NAME = "DESIRT_Bot"
TNS_BOT_ID = 105220
CATALOG_URL = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"
TNS_CSV_SKIPROWS = 1
STRIPPED_ENVELOPE_TYPE_REGEX = (
    "Ib|Ic|IIb"  # TODO: Claude notes that this also matches SLSN-1c, which may not be desired. Check.
)
MAX_STRIPPED_ENVELOPE_DISTANCE_MPC = 500


def download_tns_table():
    """Download the zipped TNS public objects CSV and return its raw bytes.

    TNS requires a registered bot: the request carries a tns_marker user-agent naming
    TNS_BOT_ID and TNS_BOT_NAME, and posts TNS_API_KEY. The catalog is regenerated daily
    after UT midnight, so repeat calls within a day fetch the same snapshot.

    Returns
    -------
    bytes
        The raw zip payload, ready for pandas.read_csv with compression="zip" and
        skiprows=TNS_CSV_SKIPROWS. That first line is the timestamp at which TNS generated
        the file, not the column header.

    Raises
    ------
    requests.HTTPError
        If TNS rejects the request, for instance on a missing or invalid API key.
    """
    user_agent = f'tns_marker{{"tns_id":"{TNS_BOT_ID}","type":"bot","name":"{TNS_BOT_NAME}"}}'
    with requests.post(
        CATALOG_URL,
        headers={"user-agent": user_agent},
        data={"api_key": (None, TNS_API_KEY)},
    ) as response:
        response.raise_for_status()
        return response.content


def clean_tns_catalog(df):
    """Filter a raw TNS catalog dataframe down to nearby stripped-envelope supernovae.

    Keeps objects whose TNS type matches STRIPPED_ENVELOPE_TYPE_REGEX, adds a luminosity
    distance for every cosmology in COSMOLOGIES, and keeps those closer than
    MAX_STRIPPED_ENVELOPE_DISTANCE_MPC under at least one of them.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw TNS public objects table. Must carry the name, ra, declination, redshift, type,
        discoverydate, reporting_group and internal_names columns; every other column is
        dropped. ra, declination, redshift and discoverydate are parsed leniently, and a row
        is dropped if any of those four fails to parse.

    Returns
    -------
    pandas.DataFrame
        The surviving rows with a reset index, carrying the eight kept columns plus one
        dist_mpc_<label> column per COSMOLOGIES entry, such as dist_mpc_Planck18.
        discoverydate is tz-aware UTC; ra, declination and redshift are numeric.

    Notes
    -----
    Redshift is taken as published, so an object recorded with z <= 0 gets a distance of
    zero or below. That passes the distance cut, so such rows survive into the result. # TODO check

    Examples
    --------
    A trimmed view of the result::

        name     ra          declination  redshift  type     dist_mpc_SHOES  dist_mpc_Planck18
        2019ebq  255.326411    -7.002923     0.037  SN Ib/c           156.2              168.5
    """
    keep_cols = [
        "name",
        "ra",
        "declination",
        "redshift",
        "type",
        "discoverydate",
        "reporting_group",
        "internal_names",
    ]
    df = df[keep_cols].copy()
    df = df[df["type"].str.contains(STRIPPED_ENVELOPE_TYPE_REGEX, na=False, regex=True)].copy()
    df["discoverydate"] = pd.to_datetime(df["discoverydate"], errors="coerce", utc=True)
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["declination"] = pd.to_numeric(df["declination"], errors="coerce")

    for label, cosmo in COSMOLOGIES.items():
        dist_col = f"dist_mpc_{label}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # TODO if possible, note why this is here.
            df[dist_col] = cosmo.luminosity_distance(df["redshift"].to_numpy()).to_value(u.Mpc)

    required = ["discoverydate", "redshift", "ra", "declination"]
    df = df.dropna(subset=required)
    near = np.zeros(len(df), dtype=bool)
    for label in COSMOLOGIES:
        near |= df[f"dist_mpc_{label}"] < MAX_STRIPPED_ENVELOPE_DISTANCE_MPC
    return df[near].reset_index(drop=True)
