"""Turn a GCN localization into a HEALPix probability map on disk.

GW notices ship a real multi-order FITS skymap, and a few of the others do too: Swift
BAT-GUANO can attach one inline, and IceCube's revised gold/bronze alerts publish one by
URL. Most GRB and neutrino notices instead give a position and an error radius, which
downstream crossmatching cannot consume directly. resolve_skymap() therefore prefers a real
map whenever the mission supplied one and only synthesizes when it did not, recording which
of the three it used so a synthesized region is never mistaken for an observed one.

Synthesized maps are written as NUNIQ multi-order FITS, the same format as the LVK maps, so
ligo.skymap.io.read_sky_map and ligo.skymap.postprocess.crossmatch read every file in the
store the same way. Multi-order is not a stylistic choice here: an Einstein Probe error
circle is about an arcminute across, and resolving that on a flat all-sky map would need
NSIDE 8192 and 5 GB per map, whereas the refined tiling below covers the same region in a
few thousand rows.
"""

import warnings
from dataclasses import replace

import astropy_healpix as ah
import healpy as hp
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from ligo.skymap import moc
from ligo.skymap.io import write_sky_map

from desi_aap.gcn_notices import notice_instruments

# How a stored map was obtained. Recorded per file in the store index, and reflected in the
# file name, because a synthesized region is a model of the mission's quoted error and not a
# measured posterior.
SKYMAP_SOURCE_INLINE = "notice_file"
SKYMAP_SOURCE_URL = "notice_url"
SKYMAP_SOURCE_SYNTHESIZED = "synthesized"

# Shape of the synthesized probability distribution. "gaussian" spreads probability as a
# 2D Gaussian calibrated so the mission's quoted radius encloses its quoted containment
# probability; "disc" spreads it uniformly inside that radius and puts nothing outside.
# Gaussian is the default because it keeps a tail beyond the quoted radius, which matters
# when the quoted error understates the true region.
SYNTHETIC_MAP_MODEL = "gaussian"
SYNTHETIC_MAP_MODEL_GAUSSIAN = "gaussian"
SYNTHETIC_MAP_MODEL_DISC = "disc"

# Refine tiles out to this many sigma from the centre. At 5 sigma the omitted tail holds
# exp(-12.5), about 4e-6 of the probability, so normalizing over the refined region loses
# nothing that matters.
SYNTHETIC_MAP_SIGMA_EXTENT = 5.0

# Target this many tiles across one sigma of the narrower axis. Three puts the 50% contour
# a few tiles from the centre, which reproduced the analytic 50% and 90% credible areas to
# better than 0.1% in testing.
SYNTHETIC_MAP_TILES_PER_SIGMA = 3.0

# The coarse tiling that covers the rest of the sky. Order 1 (48 tiles) is enough: those
# tiles hold essentially zero probability and exist only so the map tiles the whole sky,
# which keeps crossmatch from reading an uncovered target as undefined rather than excluded.
SYNTHETIC_MAP_BASE_ORDER = 1

# HEALPix order limits for the refined region. The ceiling has to clear the smallest region
# we see rather than the typical one: Einstein Probe quotes about an arcminute, and stopping
# at order 13 (26 arcsecond tiles) left only about one tile per sigma and put the recovered
# 90% area 1% off the quoted circle. Order 16 resolves 0.8 arcseconds, and because the tiling
# is multi-order the finer ceiling costs tiles only where they are needed. Order 4 is a floor
# for regions tens of degrees across.
SYNTHETIC_MAP_MIN_ORDER = 4
SYNTHETIC_MAP_MAX_ORDER = 16

# np.exp underflows to zero below about -745, so clipping here keeps far tiles at exactly
# zero density instead of raising an underflow warning.
GAUSSIAN_EXPONENT_CLIP = -700.0

# Uniform-disc density is constant inside the enclosing radius, which for the disc model is
# the radius that encloses the quoted containment probability.
DISC_INTERIOR_DENSITY = 1.0

# Radius used when a notice gives a position but no error at all. None means "do not
# synthesize", which is the honest default: BOOM's optical positions are sub-arcsecond and
# IceCube's most_probable_direction quotes no error, so inventing a radius for them would
# fabricate a region the mission never claimed.
SYNTHETIC_MAP_FALLBACK_RADIUS_DEG = None

# Fetching a mission-hosted map. The cap is a guard against a redirect to something huge,
# not a real expectation: IceCube's multi-order maps are a few MB.
REMOTE_SKYMAP_TIMEOUT_S = 30.0
REMOTE_SKYMAP_MAX_BYTES = 64 * 1024 * 1024
REMOTE_SKYMAP_CHUNK_BYTES = 1024 * 1024

# File-name suffixes. The synthesized suffix is deliberately conspicuous.
REAL_SKYMAP_SUFFIX = ".multiorder.fits"
SYNTHETIC_SKYMAP_SUFFIX = ".synthetic.multiorder.fits"

# FITS header provenance. ligo.skymap only maps header keys it knows about, so "this was
# synthesized" is carried by CREATOR plus the file name rather than a custom key. Keep the
# creator string short: a long value pushes the card over 80 columns and astropy then warns
# that it truncated the comment.
SKYMAP_ORIGIN = "CMU-MMA desi_aap"
SYNTHETIC_SKYMAP_CREATOR = "desi_aap synthetic localization"

# Extra table.meta keys describing how a synthesized map was built. Both are 8 characters and
# upper case so astropy writes them as ordinary FITS cards rather than warning and falling
# back to HIERARCH.
SYNTHETIC_MODEL_META_KEY = "SYNTHMDL"
SYNTHETIC_ORDER_META_KEY = "SYNTHORD"


def containment_radius_to_sigma(radius_deg, containment_probability):
    """Convert a quoted containment radius to the sigma of a 2D Gaussian.

    For a circular 2D Gaussian the probability inside radius r is 1 - exp(-r^2 / 2 sigma^2),
    so the radius enclosing probability p is r = sigma * sqrt(-2 ln(1 - p)). Inverting that
    is what lets a "0.5 deg at 90% containment" circle become a probability distribution
    instead of a hard edge.

    Parameters
    ----------
    radius_deg : float
        Quoted error radius, or semi-axis, in degrees.
    containment_probability : float
        Probability the quoted radius encloses, strictly between 0 and 1.

    Returns
    -------
    float
        The Gaussian sigma in degrees.

    Raises
    ------
    ValueError
        If the radius is not positive, or the containment probability is outside (0, 1).
    """
    if radius_deg is None or radius_deg <= 0:
        raise ValueError(f"error radius must be positive, got {radius_deg}")
    if not 0.0 < containment_probability < 1.0:
        raise ValueError(f"containment probability must be in (0, 1), got {containment_probability}")
    return float(radius_deg) / np.sqrt(-2.0 * np.log(1.0 - containment_probability))


def synthetic_map_order(sigma_deg):
    """Choose the HEALPix order that resolves a given sigma.

    Parameters
    ----------
    sigma_deg : float
        Gaussian sigma of the narrower axis, in degrees.

    Returns
    -------
    int
        Order whose pixels are at most sigma / SYNTHETIC_MAP_TILES_PER_SIGMA across, clamped
        to [SYNTHETIC_MAP_MIN_ORDER, SYNTHETIC_MAP_MAX_ORDER].
    """
    target_deg = float(sigma_deg) / SYNTHETIC_MAP_TILES_PER_SIGMA
    for order in range(SYNTHETIC_MAP_MIN_ORDER, SYNTHETIC_MAP_MAX_ORDER + 1):
        if hp.nside2resol(2**order, arcmin=True) / 60.0 <= target_deg:
            return order
    return SYNTHETIC_MAP_MAX_ORDER


def refine_tiles(ra_deg, dec_deg, radius_rad, max_order, base_order=SYNTHETIC_MAP_BASE_ORDER):
    """Build a non-overlapping all-sky tiling that is fine near a position and coarse elsewhere.

    Starts from a uniform base_order tiling and repeatedly splits only those tiles that
    intersect the disc, so the result stays a valid MOC: every point on the sky is covered by
    exactly one tile, and no tile contains another.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Centre of the region, in degrees.
    radius_rad : float
        Radius out to which tiles are refined, in radians.
    max_order : int
        Order the tiles inside the disc are refined to.
    base_order : int, optional
        Order of the coarse tiling covering the rest of the sky.

    Returns
    -------
    tuple of numpy.ndarray
        ``(orders, ipix)``, the HEALPix order and nested pixel index of each tile, sorted by
        order then index.
    """
    vector = hp.ang2vec(np.pi / 2 - np.deg2rad(dec_deg), np.deg2rad(ra_deg))
    tiles = {(base_order, int(index)) for index in range(hp.nside2npix(2**base_order))}
    for order in range(base_order, max_order):
        nside = 2**order
        # Grow the query radius by one tile so a tile whose centre lies outside the disc but
        # whose corner reaches into it still gets refined.
        search_radius = radius_rad + hp.max_pixrad(nside)
        intersecting = {
            int(index) for index in hp.query_disc(nside, vector, search_radius, nest=True, inclusive=True)
        }
        refined = set()
        for tile_order, tile_index in tiles:
            if tile_order == order and tile_index in intersecting:
                refined.update((order + 1, 4 * tile_index + child) for child in range(4))
            else:
                refined.add((tile_order, tile_index))
        tiles = refined
    ordered = sorted(tiles)
    orders = np.array([order for order, _ in ordered], dtype=np.int64)
    indices = np.array([index for _, index in ordered], dtype=np.int64)
    return orders, indices


def elliptical_offsets(ra_deg, dec_deg, position_angle_deg, coords):
    """Resolve tile positions into offsets along an error ellipse's axes.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Centre of the ellipse, in degrees.
    position_angle_deg : float
        Orientation of the semi-major axis in degrees, north through east.
    coords : astropy.coordinates.SkyCoord
        Tile centres.

    Returns
    -------
    tuple of numpy.ndarray
        Angular offsets in radians along the major and minor axes.
    """
    center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    separation = center.separation(coords).rad
    # position_angle() is measured north through east, the same convention gcn-schema uses
    # for ra_dec_error's third element, so the two angles subtract directly.
    delta_angle = center.position_angle(coords).rad - np.deg2rad(position_angle_deg or 0.0)
    return separation * np.cos(delta_angle), separation * np.sin(delta_angle)


def synthesize_moc_table(localization, model=SYNTHETIC_MAP_MODEL):
    """Build a multi-order probability table from a quoted error region.

    Parameters
    ----------
    localization : desi_aap.gcn_notices.Localization
        Localization with a position and a positive-size error region.
    model : str, optional
        "gaussian" or "disc"; see SYNTHETIC_MAP_MODEL.

    Returns
    -------
    astropy.table.Table
        Table with UNIQ and PROBDENSITY columns, normalized so the probability over the sky
        integrates to one, and meta describing how it was built.

    Raises
    ------
    ValueError
        If the localization has no usable error region, or the model name is unknown.
    """
    if model not in (SYNTHETIC_MAP_MODEL_GAUSSIAN, SYNTHETIC_MAP_MODEL_DISC):
        raise ValueError(f"unknown synthetic map model: {model}")
    if not localization.has_error_region:
        raise ValueError("localization has no positive-size error region to synthesize from")

    containment = localization.containment_probability
    sigma_major_deg = containment_radius_to_sigma(localization.semi_major_deg, containment)
    sigma_minor_deg = containment_radius_to_sigma(localization.semi_minor_deg, containment)

    if model == SYNTHETIC_MAP_MODEL_GAUSSIAN:
        extent_deg = SYNTHETIC_MAP_SIGMA_EXTENT * sigma_major_deg
        resolve_deg = sigma_minor_deg
    else:
        extent_deg = localization.semi_major_deg
        resolve_deg = containment_radius_to_sigma(localization.semi_minor_deg, containment)

    order = synthetic_map_order(resolve_deg)
    orders, indices = refine_tiles(localization.ra, localization.dec, np.deg2rad(extent_deg), order)
    nsides = 2**orders
    longitudes, latitudes = ah.healpix_to_lonlat(indices, nsides, order="nested")
    offset_major, offset_minor = elliptical_offsets(
        localization.ra,
        localization.dec,
        localization.position_angle_deg,
        SkyCoord(longitudes, latitudes),
    )

    if model == SYNTHETIC_MAP_MODEL_GAUSSIAN:
        exponent = -0.5 * (
            (offset_major / np.deg2rad(sigma_major_deg)) ** 2
            + (offset_minor / np.deg2rad(sigma_minor_deg)) ** 2
        )
        density = np.exp(np.clip(exponent, GAUSSIAN_EXPONENT_CLIP, 0.0))
    else:
        inside = (
            (offset_major / np.deg2rad(localization.semi_major_deg)) ** 2
            + (offset_minor / np.deg2rad(localization.semi_minor_deg)) ** 2
        ) <= 1.0
        density = np.where(inside, DISC_INTERIOR_DENSITY, 0.0)

    areas = ah.nside_to_pixel_area(nsides).to_value(u.sr)
    total = float(np.sum(density * areas))
    if not total > 0:
        raise ValueError("synthesized map has zero total probability; error region too small")
    density = density / total

    table = Table({"UNIQ": moc.nest2uniq(orders.astype(np.int8), indices), "PROBDENSITY": density})
    table.meta[SYNTHETIC_MODEL_META_KEY] = model
    table.meta[SYNTHETIC_ORDER_META_KEY] = int(order)
    return table


def write_synthetic_skymap(path, localization, record, model=SYNTHETIC_MAP_MODEL):
    """Write a synthesized multi-order skymap for one localization.

    Parameters
    ----------
    path : pathlib.Path
        Destination FITS path.
    localization : desi_aap.gcn_notices.Localization
        Localization to synthesize from.
    record : desi_aap.gcn_notices.NoticeRecord
        Notice the localization came from, used for the FITS header.
    model : str, optional
        "gaussian" or "disc"; see SYNTHETIC_MAP_MODEL.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    table = synthesize_moc_table(localization, model=model)
    object_id = record.event_id if localization.label is None else f"{record.event_id}/{localization.label}"
    header = {
        "objid": object_id,
        "creator": SYNTHETIC_SKYMAP_CREATOR,
        "origin": SKYMAP_ORIGIN,
    }
    instruments = notice_instruments(record)
    if instruments:
        header["instruments"] = instruments
    path.parent.mkdir(parents=True, exist_ok=True)
    write_sky_map(str(path), table, moc=True, nest=True, **header)
    return path


def download_skymap(url, timeout=REMOTE_SKYMAP_TIMEOUT_S, max_bytes=REMOTE_SKYMAP_MAX_BYTES):
    """Fetch a mission-hosted HEALPix map.

    Parameters
    ----------
    url : str
        URL from a notice's ``healpix_url`` field.
    timeout : float, optional
        Per-request timeout in seconds.
    max_bytes : int, optional
        Refuse a response larger than this.

    Returns
    -------
    bytes
        The downloaded file.

    Raises
    ------
    requests.RequestException
        If the request fails or returns an error status.
    ValueError
        If the response exceeds max_bytes.
    """
    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_content(REMOTE_SKYMAP_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"skymap at {url} exceeds {max_bytes} bytes")
            chunks.append(chunk)
    return b"".join(chunks)


def effective_localization(localization, fallback_radius_deg=SYNTHETIC_MAP_FALLBACK_RADIUS_DEG):
    """Apply the fallback error radius to a localization that quotes none.

    Parameters
    ----------
    localization : desi_aap.gcn_notices.Localization
        Localization as parsed from the notice.
    fallback_radius_deg : float or None, optional
        Radius in degrees to assume when the notice quotes no error region. None leaves the
        localization untouched, so no map is synthesized for it.

    Returns
    -------
    desi_aap.gcn_notices.Localization
        The localization, with semi-axes filled in if a fallback applied.
    """
    if fallback_radius_deg is None or localization.has_error_region or not localization.has_point:
        return localization
    return replace(
        localization,
        semi_major_deg=float(fallback_radius_deg),
        semi_minor_deg=float(fallback_radius_deg),
        position_angle_deg=localization.position_angle_deg or 0.0,
    )


def resolve_skymap(
    path_stem,
    localization,
    record,
    fetch_remote=True,
    model=SYNTHETIC_MAP_MODEL,
    fallback_radius_deg=SYNTHETIC_MAP_FALLBACK_RADIUS_DEG,
):
    """Write the best available skymap for one localization, preferring real maps.

    Tries, in order: the map the notice carried inline, the map it linked by URL, and
    finally a map synthesized from its quoted error region. A URL fetch that fails falls
    through to synthesis rather than losing the localization, with a warning.

    Parameters
    ----------
    path_stem : pathlib.Path
        Path without a suffix; the suffix records whether the map is real or synthesized.
    localization : desi_aap.gcn_notices.Localization
        Localization to write a map for.
    record : desi_aap.gcn_notices.NoticeRecord
        Notice the localization came from.
    fetch_remote : bool, optional
        Whether to download maps referenced by ``healpix_url``.
    model : str, optional
        Synthesis model; see SYNTHETIC_MAP_MODEL.
    fallback_radius_deg : float or None, optional
        Error radius to assume for a position that quotes none; see
        SYNTHETIC_MAP_FALLBACK_RADIUS_DEG.

    Returns
    -------
    tuple
        ``(path, source)`` where source is one of the SKYMAP_SOURCE_* constants, or
        ``(None, None)`` if the localization supports no map at all.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    if localization.healpix_bytes is not None:
        path = path_stem.with_name(path_stem.name + REAL_SKYMAP_SUFFIX)
        path.write_bytes(localization.healpix_bytes)
        return path, SKYMAP_SOURCE_INLINE

    if localization.healpix_url and fetch_remote:
        try:
            payload = download_skymap(localization.healpix_url)
        except (requests.RequestException, ValueError) as error:
            warnings.warn(
                f"could not fetch skymap for {record.event_id} from {localization.healpix_url}: {error}"
            )
        else:
            path = path_stem.with_name(path_stem.name + REAL_SKYMAP_SUFFIX)
            path.write_bytes(payload)
            return path, SKYMAP_SOURCE_URL

    usable = effective_localization(localization, fallback_radius_deg=fallback_radius_deg)
    if not usable.has_error_region:
        return None, None
    path = path_stem.with_name(path_stem.name + SYNTHETIC_SKYMAP_SUFFIX)
    write_synthetic_skymap(path, usable, record, model=model)
    return path, SKYMAP_SOURCE_SYNTHESIZED
