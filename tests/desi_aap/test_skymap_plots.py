import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from desi_aap import skymap_plots

# Every test rasterizes at this order rather than the module defaults. raster_3d_density_slice
# allocates several (npix, DISTANCE_GRID_SIZE) float64 grids, so PLOT_HEALPIX_ORDER = 8 would
# need tens of GB; order 2 is 192 pixels and runs in well under a second.
TEST_HEALPIX_ORDER = 2

SN_RA = 240.0
SN_DEC = -20.0
SN_DIST_MPC = 150.0


@pytest.fixture(autouse=True)
def agg_backend():
    """Render headlessly and close any figures a test leaves behind."""
    matplotlib.use("Agg", force=True)
    yield
    plt.close("all")


def make_moc_skymap(order=TEST_HEALPIX_ORDER, distances=True, probdensity=None):
    """Build a multiorder skymap table shaped like read_sky_map(..., moc=True) output.

    The probability is a Gaussian blob centered on the SN position, which gives
    find_greedy_credible_levels something with a real contour to trace.

    Parameters
    ----------
    order : int, optional
        HEALPix order of the single-order MOC. Defaults to TEST_HEALPIX_ORDER.
    distances : bool, optional
        If True, include the DISTMU, DISTSIGMA and DISTNORM columns the 3D slice needs.
    probdensity : numpy.ndarray, optional
        Probability density per pixel, in nested order, replacing the default blob.

    Returns
    -------
    astropy.table.Table
        Table with a UNIQ column plus PROBDENSITY and, optionally, the distance columns.
    """
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)
    if probdensity is None:
        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
        separation = np.hypot(np.degrees(phi) - SN_RA, (90 - np.degrees(theta)) - SN_DEC)
        blob = np.exp(-0.5 * np.square(separation / 25.0))
        probdensity = blob / (blob.sum() * hp.nside2pixarea(nside))
    columns = {
        "UNIQ": (4 * 4**order + np.arange(npix)).astype(np.int64),
        "PROBDENSITY": probdensity,
    }
    if distances:
        columns["DISTMU"] = np.full(npix, SN_DIST_MPC)
        columns["DISTSIGMA"] = np.full(npix, 40.0)
        columns["DISTNORM"] = np.full(npix, 1.0)
    return Table(columns)


def make_prob_skymap(prob, order=TEST_HEALPIX_ORDER):
    """Build a multiorder skymap carrying PROB rather than PROBDENSITY."""
    npix = hp.nside2npix(hp.order2nside(order))
    return Table({"UNIQ": (4 * 4**order + np.arange(npix)).astype(np.int64), "PROB": prob})


@pytest.fixture(name="row")
def row_fixture():
    """A crossmatch row shaped like one line of run_3d_spatial_crossmatch output."""
    return pd.Series(
        {
            "superevent_id": "S190425z",
            "name": "2019ebq",
            "type": "SN Ib",
            "cosmology": "Planck18",
            "ra": SN_RA,
            "declination": SN_DEC,
            "sn_dist_mpc": SN_DIST_MPC,
            "days_from_gw": 0.162,
            "searched_prob_2d": 0.21,
            "searched_prob_3d_density_rank": 0.34,
            "searched_prob_dist": 0.47,
            "credible_area_deg2": 300.0,
            "credible_volume_mpc3": 2.0e6,
            "gw_far_per_year": 1.43e-5,
            "gw_p_bns": 0.999,
            "gw_p_nsbh": 0.001,
        }
    )


@pytest.fixture(name="gw_events")
def gw_events_fixture():
    """A one-row superevent table pointing at a skymap file that need not exist."""
    return pd.DataFrame([{"superevent_id": "S190425z", "skymap_path": "S190425z__bilby.multiorder.fits"}])


def test_raster_prob_from_moc_normalizes_the_map() -> None:
    """Verify `raster_prob_from_moc` returns one probability per pixel summing to 1"""
    prob = skymap_plots.raster_prob_from_moc(make_moc_skymap(), order=TEST_HEALPIX_ORDER)

    assert prob.shape == (hp.nside2npix(hp.order2nside(TEST_HEALPIX_ORDER)),)
    assert prob.sum() == pytest.approx(1.0)
    assert (prob >= 0).all()


def test_raster_prob_from_moc_converts_probdensity_to_probability() -> None:
    """Verify `raster_prob_from_moc` multiplies PROBDENSITY by the pixel area"""
    order = TEST_HEALPIX_ORDER
    npix = hp.nside2npix(hp.order2nside(order))
    uniform = np.full(npix, 1.0 / (npix * hp.nside2pixarea(hp.order2nside(order))))

    prob = skymap_plots.raster_prob_from_moc(make_moc_skymap(probdensity=uniform), order=order)

    assert prob == pytest.approx(np.full(npix, 1.0 / npix))


def test_raster_prob_from_moc_prefers_a_prob_column() -> None:
    """Verify `raster_prob_from_moc` uses PROB directly when the skymap carries it"""
    order = TEST_HEALPIX_ORDER
    npix = hp.nside2npix(hp.order2nside(order))

    prob = skymap_plots.raster_prob_from_moc(make_prob_skymap(np.full(npix, 2.0)), order=order)

    assert prob == pytest.approx(np.full(npix, 1.0 / npix))


def test_raster_prob_from_moc_zeroes_non_finite_pixels() -> None:
    """Verify `raster_prob_from_moc` replaces NaN and infinite pixels with zero"""
    order = TEST_HEALPIX_ORDER
    npix = hp.nside2npix(hp.order2nside(order))
    values = np.full(npix, 1.0)
    values[:3] = [np.nan, np.inf, -np.inf]

    prob = skymap_plots.raster_prob_from_moc(make_prob_skymap(values), order=order)

    assert np.isfinite(prob).all()
    assert prob[:3] == pytest.approx(0.0)
    assert prob.sum() == pytest.approx(1.0)


def test_raster_prob_from_moc_leaves_an_empty_map_unnormalized() -> None:
    """Verify `raster_prob_from_moc` does not divide by zero on a map carrying no probability"""
    order = TEST_HEALPIX_ORDER
    npix = hp.nside2npix(hp.order2nside(order))

    prob = skymap_plots.raster_prob_from_moc(make_prob_skymap(np.zeros(npix)), order=order)

    assert prob.sum() == 0.0
    assert np.isfinite(prob).all()


def test_raster_3d_density_slice_returns_a_density_and_threshold() -> None:
    """Verify `raster_3d_density_slice` returns a per-pixel density and a usable threshold"""
    density, threshold = skymap_plots.raster_3d_density_slice(
        make_moc_skymap(), SN_DIST_MPC, order=TEST_HEALPIX_ORDER
    )

    assert density.shape == (hp.nside2npix(hp.order2nside(TEST_HEALPIX_ORDER)),)
    assert np.isfinite(density).all()
    assert (density >= 0).all()
    assert np.isfinite(threshold)
    assert threshold > 0
    # The threshold has to fall inside the range of the slice, or no contour can be traced.
    assert density.max() > threshold


def test_raster_3d_density_slice_peaks_at_the_distance_posterior_mean() -> None:
    """Verify `raster_3d_density_slice` evaluates the Gaussian at the distance it is given"""
    skymap = make_moc_skymap()

    at_mean, _ = skymap_plots.raster_3d_density_slice(skymap, SN_DIST_MPC, order=TEST_HEALPIX_ORDER)
    far_away, _ = skymap_plots.raster_3d_density_slice(skymap, SN_DIST_MPC + 400.0, order=TEST_HEALPIX_ORDER)

    assert at_mean.max() > far_away.max()


def test_raster_3d_density_slice_rejects_a_missing_distance() -> None:
    """Verify `raster_3d_density_slice` returns no slice when the SN distance is not finite"""
    for distance in (np.nan, np.inf):
        assert skymap_plots.raster_3d_density_slice(
            make_moc_skymap(), distance, order=TEST_HEALPIX_ORDER
        ) == (None, pytest.approx(np.nan, nan_ok=True))


def test_raster_3d_density_slice_rejects_unusable_skymaps() -> None:
    """Verify `raster_3d_density_slice` returns no slice when no pixel has a usable distance"""
    order = TEST_HEALPIX_ORDER
    npix = hp.nside2npix(hp.order2nside(order))

    no_sigma = make_moc_skymap()
    no_sigma["DISTSIGMA"] = np.nan
    density, threshold = skymap_plots.raster_3d_density_slice(no_sigma, SN_DIST_MPC, order=order)
    assert density is None
    assert np.isnan(threshold)

    no_prob = make_moc_skymap(probdensity=np.zeros(npix))
    density, threshold = skymap_plots.raster_3d_density_slice(no_prob, SN_DIST_MPC, order=order)
    assert density is None
    assert np.isnan(threshold)


def plot_a_map(skymap):
    """Draw a Mollweide map of a skymap and return its rasterized probability and axes."""
    prob = skymap_plots.raster_prob_from_moc(skymap, order=TEST_HEALPIX_ORDER)
    figure = plt.figure()
    hp.mollview(prob, nest=True, fig=figure.number)
    return prob, plt.gca()


def test_draw_contours_draws_both_contours(row) -> None:
    """Verify `draw_contours` reports drawing the 3D contour and adds lines to the active plot"""
    skymap = make_moc_skymap()
    prob, axes = plot_a_map(skymap)
    before = len(axes.lines)

    drew_3d = skymap_plots.draw_contours(prob, skymap, row, order=TEST_HEALPIX_ORDER)

    assert drew_3d is True
    assert len(axes.lines) > before
    colors = {line.get_color() for line in axes.lines}
    assert skymap_plots.PLOT_2D_CONTOUR_COLOR in colors
    assert skymap_plots.PLOT_3D_CONTOUR_COLOR in colors


def test_draw_contours_reports_an_unavailable_slice(row) -> None:
    """Verify `draw_contours` returns False but still draws the 2D contour without a 3D slice"""
    skymap = make_moc_skymap()
    prob, axes = plot_a_map(skymap)
    before = len(axes.lines)
    row["sn_dist_mpc"] = np.nan

    drew_3d = skymap_plots.draw_contours(prob, skymap, row, order=TEST_HEALPIX_ORDER)

    assert drew_3d is False
    assert len(axes.lines) > before
    colors = {line.get_color() for line in axes.lines}
    assert colors == {skymap_plots.PLOT_2D_CONTOUR_COLOR}


def install_fast_plot(monkeypatch, drew_3d=True):
    """Stub out the expensive parts of plot_3d_coincidence and record the calls made.

    read_sky_map is replaced so no FITS file is needed, and the rasterize and contour steps
    are replaced so the plot is built at TEST_HEALPIX_ORDER instead of PLOT_HEALPIX_ORDER,
    whose (npix, DISTANCE_GRID_SIZE) grids are far too large for a unit test.
    """
    skymap = make_moc_skymap()
    rasterize = skymap_plots.raster_prob_from_moc
    reads = []
    contour_calls = []

    def fake_read_sky_map(path, moc=False):
        reads.append((str(path), moc))
        return skymap

    def fake_raster_prob_from_moc(read_skymap, order=None):
        return rasterize(read_skymap, order=TEST_HEALPIX_ORDER)

    def fake_draw_contours(prob, read_skymap, plotted_row, order=None):
        contour_calls.append((prob, read_skymap, plotted_row, order))
        return drew_3d

    monkeypatch.setattr(skymap_plots, "read_sky_map", fake_read_sky_map)
    monkeypatch.setattr(skymap_plots, "raster_prob_from_moc", fake_raster_prob_from_moc)
    monkeypatch.setattr(skymap_plots, "draw_contours", fake_draw_contours)
    return reads, contour_calls


def test_plot_3d_coincidence_writes_a_plot(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` renders one file per coincidence and returns its path"""
    reads, contour_calls = install_fast_plot(monkeypatch)

    path = skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path)

    assert path == tmp_path / f"S190425z__2019ebq__Planck18.{skymap_plots.PLOT_OUTPUT_FORMAT}"
    assert path.exists()
    assert path.stat().st_size > 0
    assert reads == [("S190425z__bilby.multiorder.fits", True)]
    assert len(contour_calls) == 1
    assert plt.get_fignums() == []


def test_plot_3d_coincidence_creates_its_output_directory(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` creates the output directory when it does not exist"""
    install_fast_plot(monkeypatch)
    outdir = tmp_path / "plots"

    path = skymap_plots.plot_3d_coincidence(row, gw_events, outdir=outdir)

    assert outdir.is_dir()
    assert path.parent == outdir


def test_plot_3d_coincidence_sanitizes_the_file_name(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` runs the name parts through safe_file_part"""
    install_fast_plot(monkeypatch)
    row["name"] = "AT 2019ebq/dup"

    path = skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path)

    assert path.name == f"S190425z__AT_2019ebq_dup__Planck18.{skymap_plots.PLOT_OUTPUT_FORMAT}"
    assert path.exists()


def test_plot_3d_coincidence_keeps_the_figure_open_when_showing(
    monkeypatch, tmp_path, row, gw_events
) -> None:
    """Verify `plot_3d_coincidence` leaves the figure open for a notebook when show is True"""
    install_fast_plot(monkeypatch)

    skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path, show=True)

    assert len(plt.get_fignums()) == 1


def test_plot_3d_coincidence_uses_the_configured_figure_size(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` keeps PLOT_FIGSIZE despite healpy's warning

    hp.mollview calls pylab.figure with a figsize of its own on a figure number that
    already exists, so matplotlib returns the existing figure and healpy warns that it is
    "Ignoring specified arguments". It is healpy's 8.5x5.4 that gets ignored, not
    PLOT_FIGSIZE, and margins is passed through untouched.
    """
    install_fast_plot(monkeypatch)

    skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path, show=True)

    assert plt.gcf().get_size_inches() == pytest.approx(skymap_plots.PLOT_FIGSIZE)


def test_plot_3d_coincidence_annotates_a_missing_3d_slice(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` labels the 3D contour unavailable when none was drawn"""
    install_fast_plot(monkeypatch, drew_3d=False)

    skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path, show=True)

    labels = [text.get_text() for text in plt.gcf().legends[0].get_texts()]
    assert any("unavailable for this slice" in label for label in labels)


def test_plot_3d_coincidence_annotates_the_3d_distance(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` labels the 3D contour with the SN distance when drawn"""
    install_fast_plot(monkeypatch, drew_3d=True)

    skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path, show=True)

    labels = [text.get_text() for text in plt.gcf().legends[0].get_texts()]
    assert any(f"at {SN_DIST_MPC:.0f} Mpc" in label for label in labels)


def test_plot_3d_coincidence_tolerates_missing_event_columns(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` falls back to NaN for the optional gw_ annotation fields"""
    install_fast_plot(monkeypatch)
    row = row.drop(["gw_far_per_year", "gw_p_bns", "gw_p_nsbh"])

    path = skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path, show=True)

    assert path.exists()
    legend_text = [text.get_text() for text in plt.gcf().texts]
    assert any("FAR = nan" in text for text in legend_text)


def test_plot_3d_coincidence_requires_a_known_superevent(monkeypatch, tmp_path, row, gw_events) -> None:
    """Verify `plot_3d_coincidence` raises when the row's event is not in the event table"""
    install_fast_plot(monkeypatch)
    row["superevent_id"] = "S000000a"

    with pytest.raises(KeyError):
        skymap_plots.plot_3d_coincidence(row, gw_events, outdir=tmp_path)
