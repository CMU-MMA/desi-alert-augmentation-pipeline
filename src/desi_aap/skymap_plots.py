"""Diagnostic sky plots for SN/GW 3D coincidences.

Renders one Mollweide map per coincidence, showing the GW skymap's probability with the
SN marked and the crossmatch numbers summarized alongside, and writes it to PLOT_DIR.
"""

"""
draw_contours returning False. I wrote "the 3D slice was unavailable and only the 2D contour was drawn."
That's what the code does, but since nothing calls it, that return contract has never actually been exercised.
"""

from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ligo.skymap.io import read_sky_map
from ligo.skymap import distance, moc
from ligo.skymap.postprocess import contour, find_greedy_credible_levels
from ligo.skymap.postprocess.cosmology import dVC_dVL_for_DL

from desi_aap.gracedb_tools import CREDIBLE_LEVEL, USE_COMOVING_VOLUME_RANKING, safe_file_part

# Local output directory.
PLOT_DIR = Path("gracedb_sesn_3d_plots")

# Plot/raster settings.
PLOT_OUTPUT_FORMAT = "png"  # "pdf"
PLOT_HEALPIX_ORDER = 8
PLOT_FIGSIZE = (12, 8.5)
PLOT_MARGINS = (0.02, 0.04, 0.02, 0.12)
PLOT_PROBABILITY_MIN = 0.0
PLOT_BBOX_INCHES = "tight"
PLOT_PERCENT_SCALE = 100

# Projected 3D contour grid settings.
DISTANCE_GRID_SIZE = 1000
DISTANCE_GRID_DISTMEAN_MULTIPLIER = 6.0
DISTANCE_GRID_SN_DISTANCE_MULTIPLIER = 1.05
DISTANCE_SHELL_VARIANCE_DENOMINATOR = 12.0
GAUSSIAN_EXPONENT_FACTOR = -0.5

# HEALPix order for the projected 3D contour, deliberately coarser than PLOT_HEALPIX_ORDER.
# raster_3d_density_slice allocates several (npix, DISTANCE_GRID_SIZE) float64 grids, so cost
# rises 4x per order: order 6 needs a few GB, order 8 would need roughly 31 GB. Contours are
# smooth enough that the coarser grid costs little in the drawn result.
CONTOUR_HEALPIX_ORDER = 6

# Contour and marker styling.
PLOT_2D_CONTOUR_COLOR = "white"
PLOT_2D_CONTOUR_LINEWIDTH = 1.2
PLOT_2D_CONTOUR_ALPHA = 0.9
PLOT_3D_CONTOUR_COLOR = "#ffb000"
PLOT_3D_CONTOUR_LINEWIDTH = 1.8
PLOT_3D_CONTOUR_ALPHA = 0.95
PLOT_GRATICULE_COLOR = "0.7"
PLOT_GRATICULE_ALPHA = 0.4
SN_MARKER = "*"
SN_MARKER_SIZE = 150
SN_MARKER_COLOR = "crimson"
SN_MARKER_EDGE_COLOR = "white"
SN_MARKER_EDGE_WIDTH = 0.9
SN_MARKER_ZORDER = 10

# Contour colour-key styling. Its face is dark so the white 2D swatch stays visible.
COLOR_KEY_LOCATION = "upper left"
COLOR_KEY_FONT_SIZE = 8.5
COLOR_KEY_FACE_COLOR = "0.15"
COLOR_KEY_EDGE_COLOR = "black"
COLOR_KEY_TEXT_COLOR = "white"
COLOR_KEY_ALPHA = 0.85
COLOR_KEY_SWATCH_LENGTH = 2.6

# Plot legend box styling.
LEGEND_X = 0.035
LEGEND_Y = 0.035
LEGEND_FONT_SIZE = 8.5
LEGEND_HORIZONTAL_ALIGNMENT = "left"
LEGEND_VERTICAL_ALIGNMENT = "bottom"
LEGEND_BOX_STYLE = "round"
LEGEND_BOX_PAD = 0.45
LEGEND_FACE_COLOR = "white"
LEGEND_EDGE_COLOR = "black"
LEGEND_LINEWIDTH = 0.8
LEGEND_ALPHA = 0.88


def raster_prob_from_moc(skymap, order=PLOT_HEALPIX_ORDER):
    """Rasterize a multiorder skymap into a flat HEALPix probability-per-pixel map.

    Parameters
    ----------
    skymap : astropy.table.Table
        Multiorder skymap, as returned by ligo.skymap.io.read_sky_map with moc=True.
        Whichever of PROB or PROBDENSITY is present is used.
    order : int, optional
        HEALPix order to rasterize to. Defaults to PLOT_HEALPIX_ORDER.

    Returns
    -------
    numpy.ndarray
        Probability per pixel in nested ordering, with non-finite values zeroed and the
        total normalized to 1. A map summing to zero is returned unnormalized.
    """
    raster = moc.rasterize(skymap, order=order)
    nside = hp.npix2nside(len(raster))
    if "PROB" in raster.dtype.names:
        prob = np.asarray(raster["PROB"], dtype=float)
    else:
        prob = np.asarray(raster["PROBDENSITY"], dtype=float) * hp.nside2pixarea(nside)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    total = prob.sum()
    if total > 0:
        prob = prob / total
    return prob


def raster_3d_density_slice(skymap, dl_mpc, contour_level=CREDIBLE_LEVEL, order=PLOT_HEALPIX_ORDER):
    """Return an approximate projected 3D contour map at one distance.


    Parameters
    ----------
    skymap : astropy.table.Table
        Multiorder skymap carrying PROBDENSITY, DISTMU, DISTSIGMA and DISTNORM columns.
    dl_mpc : float
        Luminosity distance in Mpc to evaluate at, normally the SN's distance under the
        cosmology being plotted.
    contour_level : float, optional
        Credible level the threshold should enclose. Defaults to CREDIBLE_LEVEL.
    order : int, optional
        HEALPix order to rasterize to. Defaults to PLOT_HEALPIX_ORDER.

    Returns
    -------
    density_at_distance : numpy.ndarray or None
        Per-pixel posterior density at dl_mpc, in nested ordering with non-finite values
        zeroed, directly comparable against density_threshold. None when the skymap has no
        usable distance pixels, when dl_mpc is not finite, or when the posterior carries no
        probability.
    density_threshold : float
        Density bounding the contour_level region, or NaN whenever density_at_distance is
        None.
    """
    raster = moc.rasterize(skymap, order=order)
    nside = hp.npix2nside(len(raster))
    dA = np.full(len(raster), hp.nside2pixarea(nside))

    dP_dA = np.asarray(raster["PROBDENSITY"], dtype=float)
    mu = np.asarray(raster["DISTMU"], dtype=float)
    sigma = np.asarray(raster["DISTSIGMA"], dtype=float)
    norm = np.asarray(raster["DISTNORM"], dtype=float)

    valid = (
        np.isfinite(dP_dA)
        & np.isfinite(mu)
        & np.isfinite(sigma)
        & np.isfinite(norm)
        & (dP_dA > 0)
        & (sigma > 0)
        & (norm > 0)
    )
    if not valid.any() or not np.isfinite(dl_mpc):
        return None, np.nan

    dP = np.zeros_like(dP_dA)
    dP[valid] = dP_dA[valid] * dA[valid]
    distmean, _ = distance.parameters_to_marginal_moments(dP[valid], mu[valid], sigma[valid])

    max_r = max(
        DISTANCE_GRID_DISTMEAN_MULTIPLIER * distmean,
        DISTANCE_GRID_SN_DISTANCE_MULTIPLIER * dl_mpc,
    )
    if not np.isfinite(max_r) or max_r <= 0:
        return None, np.nan
    d_r = max_r / DISTANCE_GRID_SIZE
    r = d_r * np.arange(1, DISTANCE_GRID_SIZE)

    dV = (np.square(r) + np.square(d_r) / DISTANCE_SHELL_VARIANCE_DENOMINATOR) * d_r * dA.reshape(-1, 1)
    radial_density = np.exp(
        GAUSSIAN_EXPONENT_FACTOR * np.square((r.reshape(1, -1) - mu.reshape(-1, 1)) / sigma.reshape(-1, 1))
    ) * (dP_dA * norm / (sigma * np.sqrt(2 * np.pi))).reshape(-1, 1)
    dP_grid = radial_density * dV
    dP_grid[~np.isfinite(dP_grid)] = 0

    rank_volume = dV
    if USE_COMOVING_VOLUME_RANKING:
        rank_volume = dV * dVC_dVL_for_DL(r).reshape(1, -1)
    density_grid = dP_grid / rank_volume
    density_grid[~np.isfinite(density_grid)] = 0

    order_idx = np.flipud(np.argsort(density_grid.ravel()))
    ranked_prob = dP_grid.ravel()[order_idx]
    ranked_density = density_grid.ravel()[order_idx]
    cumulative_prob = np.cumsum(ranked_prob)
    if cumulative_prob.size == 0 or cumulative_prob[-1] <= 0:
        return None, np.nan

    target_prob = contour_level * cumulative_prob[-1]
    threshold_idx = min(np.searchsorted(cumulative_prob, target_prob), len(ranked_density) - 1)
    density_threshold = ranked_density[threshold_idx]

    density_at_distance = np.exp(GAUSSIAN_EXPONENT_FACTOR * np.square((dl_mpc - mu) / sigma)) * (
        dP_dA * norm / (sigma * np.sqrt(2 * np.pi))
    )
    if USE_COMOVING_VOLUME_RANKING:
        density_at_distance = density_at_distance / dVC_dVL_for_DL(dl_mpc)
    density_at_distance = np.nan_to_num(density_at_distance, nan=0.0, posinf=0.0, neginf=0.0)
    return density_at_distance, density_threshold


def draw_contours(prob, skymap, row, order=CONTOUR_HEALPIX_ORDER):
    """Overlay the 2D and projected 3D credible contours on the current healpy plot.

    Draws into the active matplotlib figure through healpy's projection helpers, so a map
    must already have been plotted. The 2D contour is always attempted; the 3D one is
    projected at the SN's distance and may be unavailable for that slice.

    Parameters
    ----------
    prob : numpy.ndarray
        Probability per pixel, from raster_prob_from_moc.
    skymap : astropy.table.Table
        The multiorder skymap those probabilities came from, needed for its distance
        columns.
    row : pandas.Series
        A row of the run_3d_spatial_crossmatch output. Only sn_dist_mpc is read, giving the
        distance the 3D contour is projected at.
    order : int, optional
        HEALPix order for the 3D slice. Defaults to CONTOUR_HEALPIX_ORDER, which is coarser
        than the map itself because the slice grid's cost rises 4x per order.

    Returns
    -------
    bool
        True if at least one 3D contour polygon was drawn, False if the 3D slice was
        unavailable and only the 2D contour was drawn.
    """
    credible_2d = find_greedy_credible_levels(prob)
    for polygon in contour(credible_2d, [CREDIBLE_LEVEL], nest=True, degrees=True)[0]:
        polygon = np.asarray(polygon)
        if len(polygon):
            hp.projplot(
                polygon[:, 0],
                polygon[:, 1],
                lonlat=True,
                color=PLOT_2D_CONTOUR_COLOR,
                linewidth=PLOT_2D_CONTOUR_LINEWIDTH,
                alpha=PLOT_2D_CONTOUR_ALPHA,
            )

    density_slice, density_threshold = raster_3d_density_slice(
        skymap, row["sn_dist_mpc"], CREDIBLE_LEVEL, order=order
    )
    if density_slice is None or not np.isfinite(density_threshold) or density_threshold <= 0:
        return False

    drew_3d = False
    for polygon in contour(density_slice, [density_threshold], nest=True, degrees=True)[0]:
        polygon = np.asarray(polygon)
        if len(polygon):
            hp.projplot(
                polygon[:, 0],
                polygon[:, 1],
                lonlat=True,
                color=PLOT_3D_CONTOUR_COLOR,
                linewidth=PLOT_3D_CONTOUR_LINEWIDTH,
                alpha=PLOT_3D_CONTOUR_ALPHA,
            )
            drew_3d = True
    return drew_3d


def plot_3d_coincidence(row, gw_events, outdir=PLOT_DIR, show=False, output_format=PLOT_OUTPUT_FORMAT):
    """Render and save a Mollweide sky plot for one SN/GW coincidence.

    Plots the event's skymap probability, marks the SN, and annotates the figure with the
    crossmatch numbers for that SN and cosmology. One file is written per call, so callers
    iterating over coincidences get one plot per (SN, event, cosmology) row.

    Parameters
    ----------
    row : pandas.Series
        A row of the run_3d_spatial_crossmatch output, normally one of the coincidences
        filtered on spatial_status and inside_3d_credible_level. Read for superevent_id,
        name, type, cosmology, ra, declination, sn_dist_mpc, days_from_gw,
        searched_prob_2d, searched_prob_3d_density_rank, searched_prob_dist,
        credible_area_deg2 and credible_volume_mpc3, plus gw_far_per_year, gw_p_bns and
        gw_p_nsbh, which fall back to NaN in the annotation when absent.
    gw_events : pandas.DataFrame
        Superevent table from fetch_gracedb_superevents, used to look up the event's
        skymap_path by superevent_id.
    outdir : pathlib.Path, optional
        Directory to write into, created if it does not exist. Defaults to PLOT_DIR, which
        is relative to the working directory.
    show : bool, optional
        If True, leave the figure open so a notebook's inline backend renders it. The file
        is written either way. Defaults to False, which closes the figure; leaving figures
        open across many coincidences accumulates them in memory.
    output_format : str, optional
        File extension to write, without a leading dot, such as "png" or "pdf". Any format
        the active matplotlib backend supports is accepted, and it is the extension that
        selects it. Defaults to PLOT_OUTPUT_FORMAT.

    Returns
    -------
    pathlib.Path
        Path to the written file, named
        "<superevent_id>__<name>__<cosmology>.<output_format>".

    Raises
    ------
    ValueError
        From matplotlib, if output_format is not a format the active backend can write.
    """
    outdir.mkdir(exist_ok=True)
    event = gw_events.set_index("superevent_id").loc[row["superevent_id"]]
    skymap = read_sky_map(event["skymap_path"], moc=True)
    prob = raster_prob_from_moc(skymap)

    title = f"{row['superevent_id']} / {row['name']} ({row['type']}) / {row['cosmology']}"
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    hp.mollview(
        prob,
        nest=True,
        fig=fig.number,
        title=title,
        unit="probability per pixel",
        cmap="viridis",
        min=PLOT_PROBABILITY_MIN,
        cbar=True,
        margins=PLOT_MARGINS,
    )
    hp.graticule(color=PLOT_GRATICULE_COLOR, alpha=PLOT_GRATICULE_ALPHA)

    drew_3d = draw_contours(prob, skymap, row)

    hp.projscatter(
        row["ra"],
        row["declination"],
        lonlat=True,
        marker=SN_MARKER,
        s=SN_MARKER_SIZE,
        color=SN_MARKER_COLOR,
        edgecolors=SN_MARKER_EDGE_COLOR,
        linewidths=SN_MARKER_EDGE_WIDTH,
        label=row["name"],
        zorder=SN_MARKER_ZORDER,
    )

    percent = PLOT_PERCENT_SCALE * CREDIBLE_LEVEL
    three_d_label = (
        f"3D density-rank {percent:.0f}% at {row['sn_dist_mpc']:.0f} Mpc"
        if drew_3d
        else f"3D density-rank {percent:.0f}% (unavailable for this slice)"
    )
    color_key = fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=PLOT_2D_CONTOUR_COLOR,
                linewidth=PLOT_2D_CONTOUR_LINEWIDTH,
                label=f"2D sky {percent:.0f}%",
            ),
            Line2D(
                [],
                [],
                color=PLOT_3D_CONTOUR_COLOR,
                linewidth=PLOT_3D_CONTOUR_LINEWIDTH,
                label=three_d_label,
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker=SN_MARKER,
                markerfacecolor=SN_MARKER_COLOR,
                markeredgecolor=SN_MARKER_EDGE_COLOR,
                markeredgewidth=SN_MARKER_EDGE_WIDTH,
                label=row["name"],
            ),
        ],
        loc=COLOR_KEY_LOCATION,
        fontsize=COLOR_KEY_FONT_SIZE,
        facecolor=COLOR_KEY_FACE_COLOR,
        edgecolor=COLOR_KEY_EDGE_COLOR,
        framealpha=COLOR_KEY_ALPHA,
        labelcolor=COLOR_KEY_TEXT_COLOR,
        handlelength=COLOR_KEY_SWATCH_LENGTH,
    )
    color_key.get_frame().set_linewidth(LEGEND_LINEWIDTH)

    legend_text = (
        f"delay = {row['days_from_gw']:+.2f} d\n"
        f"FAR = {row.get('gw_far_per_year', np.nan):.3g} yr^-1\n"
        f"p_NSNS = {row.get('gw_p_bns', np.nan):.3g}    p_NSBH = {row.get('gw_p_nsbh', np.nan):.3g}\n"
        f"D_L({row['cosmology']}) = {row['sn_dist_mpc']:.1f} Mpc\n"
        f"2D sky CL = {row['searched_prob_2d']:.3f}\n"
        f"3D density-rank = {row['searched_prob_3d_density_rank']:.3f}\n"
        f"distance CDF = {row['searched_prob_dist']:.3f}\n"
        f"credible area = {row['credible_area_deg2']:.1f} deg^2\n"
        f"credible 3D volume = {row['credible_volume_mpc3']:.3g} Mpc^3"
    )
    fig.text(
        LEGEND_X,
        LEGEND_Y,
        legend_text,
        fontsize=LEGEND_FONT_SIZE,
        ha=LEGEND_HORIZONTAL_ALIGNMENT,
        va=LEGEND_VERTICAL_ALIGNMENT,
        bbox={
            "boxstyle": f"{LEGEND_BOX_STYLE},pad={LEGEND_BOX_PAD}",
            "facecolor": LEGEND_FACE_COLOR,
            "edgecolor": LEGEND_EDGE_COLOR,
            "linewidth": LEGEND_LINEWIDTH,
            "alpha": LEGEND_ALPHA,
        },
    )

    filename = (
        f"{safe_file_part(row['superevent_id'])}__"
        f"{safe_file_part(row['name'])}__"
        f"{safe_file_part(row['cosmology'])}.{str(output_format).lstrip('.')}"
    )
    path = outdir / filename
    fig.savefig(path, bbox_inches=PLOT_BBOX_INCHES)
    if not show:
        plt.close(fig)
    return path
