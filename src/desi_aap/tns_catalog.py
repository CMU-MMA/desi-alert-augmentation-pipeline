"""Download and filter the TNS public catalog for stripped-envelope supernovae (SESN).

The Transient Name Server publishes its whole public object catalog as a zipped CSV that is
regenerated daily after UT midnight. Downloading it requires a registered TNS bot, whose
credentials are read from the environment; see tns_credentials.
"""

import os

import numpy as np
import pandas as pd
import requests
from astropy import units as u

from desi_aap.cosmology import COSMOLOGIES

# Names of the environment variables holding the TNS bot credentials. They are read at call
# time rather than at import, so the package imports fine without them and only
# download_tns_table requires them to be set. In CI they come from repository secrets of the
# same names; locally, export them in the shell or notebook kernel first.
TNS_API_KEY_ENV = "TNS_API_KEY"
TNS_BOT_ID_ENV = "TNS_BOT_ID"
TNS_BOT_NAME_ENV = "TNS_BOT_NAME"

# TNS settings copied from the existing workflow.
CATALOG_URL = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"
TNS_CSV_SKIPROWS = 1
STRIPPED_ENVELOPE_TYPE_REGEX = (
    "Ib|Ic|IIb"  # TODO: Claude notes that this also matches SLSN-1c, which may not be desired. Check.
)
MAX_STRIPPED_ENVELOPE_DISTANCE_MPC = 500
# Smallest redshift treated as a real measurement. Roughly 0.9 Mpc under either
# cosmology, so it excludes the Local Group along with the z <= 0 placeholders.
MIN_REDSHIFT = 0.0002


def tns_credentials():
    """Read the TNS bot credentials from the environment.

    Kept separate from download_tns_table so the "are we configured?" check can be made
    without issuing a request, and so a misconfiguration reports which variable is at fault
    rather than surfacing as an opaque HTTP error from TNS.

    Returns
    -------
    api_key : str
        Value of $TNS_API_KEY.
    bot_id : str
        Value of $TNS_BOT_ID, validated as an integer but returned as a string, which is how
        it is embedded in the tns_marker user-agent.
    bot_name : str
        Value of $TNS_BOT_NAME.

    Raises
    ------
    RuntimeError
        If any of the three variables is unset or empty, or if $TNS_BOT_ID is not an
        integer. The latter most often means two of the values were swapped.
    """
    names = (TNS_API_KEY_ENV, TNS_BOT_ID_ENV, TNS_BOT_NAME_ENV)
    values = {name: (os.environ.get(name) or "").strip() for name in names}

    missing = [name for name in names if not values[name]]
    if missing:
        raise RuntimeError(
            f"TNS bot credentials are not configured: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} unset or empty. Set "
            f"{', '.join(names)} in the environment; in CI they come from the repository "
            "secrets of the same names."
        )

    bot_id = values[TNS_BOT_ID_ENV]
    try:
        int(bot_id)
    except ValueError:
        raise RuntimeError(
            f"{TNS_BOT_ID_ENV} must be an integer TNS bot id, got {bot_id!r}. Check that "
            f"{TNS_BOT_ID_ENV} and {TNS_BOT_NAME_ENV} are not swapped."
        ) from None

    return values[TNS_API_KEY_ENV], bot_id, values[TNS_BOT_NAME_ENV]


def download_tns_table():
    """Download the zipped TNS public objects CSV and return its raw bytes.

    TNS requires a registered bot: the request carries a tns_marker user-agent naming the
    bot id and name, and posts the API key. All three come from the environment via
    tns_credentials. The catalog is regenerated daily after UT midnight, so repeat calls
    within a day fetch the same snapshot.

    Returns
    -------
    bytes
        The raw zip payload, ready for pandas.read_csv with compression="zip" and
        skiprows=TNS_CSV_SKIPROWS. That first line is the timestamp at which TNS generated
        the file, not the column header.

    Raises
    ------
    RuntimeError
        If the TNS bot credentials are not configured; see tns_credentials.
    requests.HTTPError
        If TNS rejects the request, for instance on an invalid API key.
    """
    api_key, bot_id, bot_name = tns_credentials()
    user_agent = f'tns_marker{{"tns_id":"{bot_id}","type":"bot","name":"{bot_name}"}}'
    with requests.post(
        CATALOG_URL,
        headers={"user-agent": user_agent},
        data={"api_key": (None, api_key)},
    ) as response:
        response.raise_for_status()
        return response.content


def clean_tns_catalog(df):
    """Filter a raw TNS catalog dataframe down to nearby stripped-envelope supernovae.

    Keeps objects whose TNS type matches STRIPPED_ENVELOPE_TYPE_REGEX and whose redshift is
    at least MIN_REDSHIFT, adds a luminosity distance for every cosmology in COSMOLOGIES,
    and keeps those closer than MAX_STRIPPED_ENVELOPE_DISTANCE_MPC under at least one of
    them.

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
    Redshift is floored at MIN_REDSHIFT rather than taken as published, because neither of
    the two ways a catalog redshift can be non-positive yields a meaningful luminosity
    distance. A z of exactly 0 places an extragalactic transient at zero distance, which is
    not a measurement however it arose. A negative z is a genuine blueshift, which happens
    for very nearby hosts whose peculiar velocity outruns the Hubble flow, but is not a
    distance either. The floor is applied before the cosmology loop, so neither kind of row
    reaches astropy at all.

    How TNS represents an unmeasured redshift does not matter here, which is why it is not
    documented above: an explicit 0 fails the floor comparison directly, and a blank becomes
    NaN under pd.to_numeric, which fails every comparison and so is dropped at the same line.
    Both spellings leave by the same door. This was worth knowing only before the floor
    existed, when a 0 survived to the output and a NaN did not.

    It is not a cosmetic cut. Without it the failure modes are uneven, and the one that
    matters is reachable in practice, since realistically blueshifted hosts land near
    z = -0.001, inside the band that raises::

        redshift        distance  outcome if the floor were removed
        -------------  ---------  ----------------------------------------------------------
        z == 0           0.0 Mpc  no crash; SN sits at the origin, credible level meaningless
        -1 < z < 0      negative   ValueError from SkyCoord in run_3d_spatial_crossmatch
        z == -1         -0.0 Mpc  no crash; -0.0 >= 0 in IEEE 754, so SkyCoord accepts it
        z < -1                 -   Planck18 raises TypeError on astropy 7.1.1, returns NaN
                                   on 5.3, and raises ZeroDivisionError at z == -2 on both

    A NaN redshift, left by a value pd.to_numeric could not parse, also fails the floor
    comparison, so it is dropped there rather than by the dropna further down.

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
    # Ahead of the cosmology loop rather than alongside the other cuts below, so that the
    # redshifts with no usable luminosity distance never reach astropy. NaN fails this
    # comparison too, so an unparseable redshift is dropped here.
    df = df[df["redshift"] >= MIN_REDSHIFT]

    for label, cosmo in COSMOLOGIES.items():
        dist_col = f"dist_mpc_{label}"
        if df.empty:
            # Planck18.luminosity_distance goes through np.vectorize, which rejects a
            # size-0 array with "cannot call vectorize on size 0 inputs unless otypes is
            # set" rather than returning an empty result. Assigning the empty column
            # directly keeps a catalog with no stripped-envelope objects, or an empty input,
            # from raising ValueError out of this loop.
            df[dist_col] = np.array([], dtype=float)
            continue
        df[dist_col] = cosmo.luminosity_distance(df["redshift"].to_numpy()).to_value(u.Mpc)

    required = ["discoverydate", "redshift", "ra", "declination"]
    df = df.dropna(subset=required)
    near = np.zeros(len(df), dtype=bool)
    for label in COSMOLOGIES:
        near |= df[f"dist_mpc_{label}"] < MAX_STRIPPED_ENVELOPE_DISTANCE_MPC
    return df[near].reset_index(drop=True)
