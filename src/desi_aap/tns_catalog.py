"""Download and filter the TNS public catalog for stripped-envelope supernovae (SESN)."""

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
STRIPPED_ENVELOPE_TYPE_REGEX = "Ib|Ic|IIb"
MAX_STRIPPED_ENVELOPE_DISTANCE_MPC = 500


def download_tns_table():
    """Download the zipped TNS public objects CSV and return its raw bytes."""
    user_agent = f'tns_marker{{"tns_id":"{TNS_BOT_ID}","type":"bot","name":"{TNS_BOT_NAME}"}}'
    with requests.post(
        CATALOG_URL,
        headers={"user-agent": user_agent},
        data={"api_key": (None, TNS_API_KEY)},
    ) as response:
        response.raise_for_status()
        return response.content


def clean_tns_catalog(df):
    """Filter a raw TNS catalog dataframe down to nearby stripped-envelope supernovae."""
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
            warnings.simplefilter("ignore", RuntimeWarning)
            df[dist_col] = cosmo.luminosity_distance(df["redshift"].to_numpy()).to_value(u.Mpc)

    required = ["discoverydate", "redshift", "ra", "declination"]
    df = df.dropna(subset=required)
    near = np.zeros(len(df), dtype=bool)
    for label in COSMOLOGIES:
        near |= df[f"dist_mpc_{label}"] < MAX_STRIPPED_ENVELOPE_DISTANCE_MPC
    return df[near].reset_index(drop=True)
