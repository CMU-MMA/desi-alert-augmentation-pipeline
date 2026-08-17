"""Download and filter the TNS public catalog for stripped-envelope supernovae (SESN).

The Transient Name Server publishes its whole public object catalog as a zipped CSV that is
regenerated daily after UT midnight. Downloading it requires a registered TNS bot, whose
credentials are read from the environment; see tns_credentials.
"""

import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from astropy import units as u

from desi_aap.cosmology import COSMOLOGIES


def tns_credentials():
    """Read the TNS bot credentials from the environment.

    Kept separate from download_tns_table so the "are we configured?" check can be made
    without issuing a request, and so a misconfiguration reports which variable is at fault
    rather than surfacing as an opaque HTTP error from TNS.

    The three variable names are fixed rather than configurable: they are read at call time
    rather than at import, so the package imports fine without them and only
    download_tns_table requires them to be set. In CI they come from repository secrets of
    the same names; locally, export them in the shell or notebook kernel first.

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
    api_key_env = "TNS_API_KEY"
    bot_id_env = "TNS_BOT_ID"
    bot_name_env = "TNS_BOT_NAME"

    names = (api_key_env, bot_id_env, bot_name_env)
    values = {name: (os.environ.get(name) or "").strip() for name in names}

    missing = [name for name in names if not values[name]]
    if missing:
        raise RuntimeError(
            f"TNS bot credentials are not configured: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} unset or empty. Set "
            f"{', '.join(names)} in the environment; in CI they come from the repository "
            "secrets of the same names."
        )

    bot_id = values[bot_id_env]
    try:
        int(bot_id)
    except ValueError:
        raise RuntimeError(
            f"{bot_id_env} must be an integer TNS bot id, got {bot_id!r}. Check that "
            f"{bot_id_env} and {bot_name_env} are not swapped."
        ) from None

    return values[api_key_env], bot_id, values[bot_name_env]


def download_tns_table():
    """Download the TNS public objects catalog and return it as a dataframe.

    TNS requires a registered bot: the request carries a tns_marker user-agent naming the
    bot id and name, and posts the API key. All three come from the environment via
    tns_credentials. The catalog is regenerated daily after UT midnight, so repeat calls
    within a day fetch the same snapshot.

    The payload arrives as a zipped CSV whose first line is the timestamp at which TNS
    generated the file rather than the column header, so it is skipped. That detail is
    handled here, next to the request that knows the format, rather than being left to
    callers.

    Returns
    -------
    pandas.DataFrame
        The catalog as published, every column still a string unless pandas inferred
        otherwise, ready for clean_tns_catalog.

    Raises
    ------
    RuntimeError
        If the TNS bot credentials are not configured; see tns_credentials.
    requests.HTTPError
        If TNS rejects the request, for instance on an invalid API key.
    """
    catalog_url = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"
    # The generation timestamp TNS writes above the header row.
    tns_csv_skiprows = 1

    api_key, bot_id, bot_name = tns_credentials()
    user_agent = f'tns_marker{{"tns_id":"{bot_id}","type":"bot","name":"{bot_name}"}}'
    with requests.post(
        catalog_url,
        headers={"user-agent": user_agent},
        data={"api_key": (None, api_key)},
    ) as response:
        response.raise_for_status()
        payload = response.content

    return pd.read_csv(BytesIO(payload), skiprows=tns_csv_skiprows, compression="zip", low_memory=False)


def clean_tns_catalog(
    df,
    *,
    # TODO: Claude notes that this also matches SLSN-1c, which may not be desired. Check.
    stripped_env_type_regex="Ib|Ic|IIb",
    max_stripped_env_distance_mpc=500,
    min_redshift=0.0002,
):
    """Filter a raw TNS catalog dataframe down to nearby stripped-envelope supernovae.

    Keeps objects whose TNS type matches stripped_env_type_regex and whose redshift is at
    least min_redshift, adds a luminosity distance for every cosmology in COSMOLOGIES, and
    keeps those closer than max_stripped_env_distance_mpc under at least one of them.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw TNS public objects table, as returned by download_tns_table. Must carry the
        name, ra, declination, redshift, type, discoverydate, reporting_group and
        internal_names columns; every other column is dropped. ra, declination, redshift and
        discoverydate are parsed leniently, and a row is dropped if any of those four fails
        to parse.
    stripped_env_type_regex : str, optional
        Regex matched against the TNS type column to select stripped-envelope supernovae.
        Defaults to "Ib|Ic|IIb".
    max_stripped_env_distance_mpc : float, optional
        Luminosity-distance cut in Mpc, exclusive. An object is kept if it falls inside this
        under at least one cosmology, not all of them. Defaults to 500.
    min_redshift : float, optional
        Smallest redshift treated as a real measurement, inclusive. Roughly 0.9 Mpc under
        either cosmology, so it excludes the Local Group along with the z <= 0 placeholders;
        see the Notes below for why the floor exists at all. Defaults to 0.0002.

    Returns
    -------
    pandas.DataFrame
        The surviving rows with a reset index, carrying the eight kept columns plus one
        dist_mpc_<label> column per COSMOLOGIES entry, such as dist_mpc_Planck18.
        discoverydate is tz-aware UTC; ra, declination and redshift are numeric.

    Notes
    -----
    Redshift is floored at min_redshift rather than taken as published, because neither of
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
    df = df[df["type"].str.contains(stripped_env_type_regex, na=False, regex=True)].copy()
    df["discoverydate"] = pd.to_datetime(df["discoverydate"], errors="coerce", utc=True)
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["declination"] = pd.to_numeric(df["declination"], errors="coerce")
    # Ahead of the cosmology loop rather than alongside the other cuts below, so that the
    # redshifts with no usable luminosity distance never reach astropy. NaN fails this
    # comparison too, so an unparseable redshift is dropped here.
    df = df[df["redshift"] >= min_redshift]

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
        near |= df[f"dist_mpc_{label}"] < max_stripped_env_distance_mpc
    return df[near].reset_index(drop=True)
