"""Download and filter the TNS public catalog for stripped-envelope supernovae (SESN).

The Transient Name Server publishes its whole public object catalog as a zipped CSV that is
regenerated daily after UT midnight. Downloading it requires a registered TNS bot, whose
credentials are read from the environment; see tns_credentials.
"""

import os
import warnings

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
    # TODO: Check the following with Xander.

    Redshift is taken as published and is not required to be positive. TNS carries z = 0
    where no redshift was measured, and genuinely negative z for very nearby hosts whose
    peculiar velocity outruns the Hubble flow. Neither is filtered here, so the result can
    carry distances of zero or below, which are not meaningful as luminosity distances.

    The regimes do not degrade uniformly. Realistically blueshifted hosts land near
    z = -0.001, inside the only band that raises from run_3d_spatial_crossmatch::

        redshift        distance  outcome
        -------------  ---------  ----------------------------------------------------------
        z > 0           positive  fine
        z == 0           0.0 Mpc  no crash; SN sits at the origin, credible level meaningless
        -1 < z < 0      negative   ValueError from SkyCoord in run_3d_spatial_crossmatch
        z == -1         -0.0 Mpc  no crash; -0.0 >= 0 in IEEE 754, so SkyCoord accepts it
        z < -1                 -   no usable distance; the row cannot survive, see below

    That last row is where the cosmologies part company, and it is Planck18 that decides it
    for the function as a whole, since every COSMOLOGIES entry is evaluated. What Planck18
    does below z = -1 depends on the astropy version: on 7.1.1 its integrand goes complex
    and luminosity_distance raises TypeError, aborting this function, while on 5.3 it
    returns NaN and the row is instead dropped by the distance cut, NaN < MAX comparing
    False. Both versions raise ZeroDivisionError at z == -2 exactly. SHOES never raises
    anywhere below -1 on either: it returns a positive, unphysical distance down to about
    z = -2.4 and NaN below that. So such a row is either fatal or filtered, never returned,
    but it would be silently kept at a meaningless distance if Planck18 were ever dropped
    from COSMOLOGIES.

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
        if df.empty:
            # Planck18.luminosity_distance goes through np.vectorize, which rejects a
            # size-0 array with "cannot call vectorize on size 0 inputs unless otypes is
            # set" rather than returning an empty result. Assigning the empty column
            # directly keeps a catalog with no stripped-envelope objects, or an empty input,
            # from raising ValueError out of this loop.
            df[dist_col] = np.array([], dtype=float)
            continue
        with warnings.catch_warnings():
            # Live, but only in one corner: on astropy 7.1.1 the only RuntimeWarning
            # luminosity_distance could be provoked into raising is SHOES below about
            # z = -2.4, where the comoving-distance integral fails and returns NaN. Note
            # that scipy's IntegrationWarning, emitted alongside it there, is a UserWarning
            # and passes straight through this filter, and that Planck18 raises TypeError
            # anywhere below z = -1, which this would not catch either.
            # TODO Check with Xander.
            warnings.simplefilter("ignore", RuntimeWarning)
            df[dist_col] = cosmo.luminosity_distance(df["redshift"].to_numpy()).to_value(u.Mpc)

    required = ["discoverydate", "redshift", "ra", "declination"]
    df = df.dropna(subset=required)
    near = np.zeros(len(df), dtype=bool)
    for label in COSMOLOGIES:
        near |= df[f"dist_mpc_{label}"] < MAX_STRIPPED_ENVELOPE_DISTANCE_MPC
    return df[near].reset_index(drop=True)
