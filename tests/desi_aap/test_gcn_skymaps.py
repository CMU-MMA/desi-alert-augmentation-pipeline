"""Tests for resolving and synthesizing GCN localization maps."""

import astropy_healpix as ah
import numpy as np
import pytest
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from gcn_examples import build_moc_fits_bytes, icecube_gold_bronze, swift_bat_guano
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch

from desi_aap import gcn_notices, gcn_skymaps

# A half-degree circle at 90% containment, the shape Swift GUANO and IceCube both quote.
TEST_RA = 336.26
TEST_DEC = 25.139
TEST_RADIUS_DEG = 0.5
TEST_CONTAINMENT = 0.9

# The synthesized map is a discretization of an analytic Gaussian, so its credible areas
# should match the closed form to well under a percent. Testing measured 0.05%.
AREA_TOLERANCE = 0.01

# 4*pi steradians in square degrees; a valid MOC tiles the whole sky exactly once.
FULL_SKY_SR = 4.0 * np.pi


def circle_localization(radius_deg=TEST_RADIUS_DEG, containment=TEST_CONTAINMENT, ra=TEST_RA, dec=TEST_DEC):
    """Build a circular Localization for the synthesis tests.

    Parameters
    ----------
    radius_deg : float, optional
        Quoted error radius in degrees.
    containment : float, optional
        Probability the radius encloses.
    ra, dec : float, optional
        Centre in degrees.

    Returns
    -------
    desi_aap.gcn_notices.Localization
        The localization.
    """
    return gcn_notices.Localization(
        ra=ra,
        dec=dec,
        semi_major_deg=radius_deg,
        semi_minor_deg=radius_deg,
        position_angle_deg=0.0,
        containment_probability=containment,
    )


def gaussian_credible_area_deg2(radius_deg, containment, level):
    """Return the analytic credible area of a circular Gaussian, in square degrees.

    A circular 2D Gaussian encloses probability ``level`` inside radius
    ``sigma * sqrt(-2 ln(1 - level))``, so its credible area is pi times that radius squared.

    Parameters
    ----------
    radius_deg : float
        Quoted error radius.
    containment : float
        Probability the quoted radius encloses.
    level : float
        Credible level to compute the area for.

    Returns
    -------
    float
        Area in square degrees.
    """
    sigma = gcn_skymaps.containment_radius_to_sigma(radius_deg, containment)
    return np.pi * (sigma * np.sqrt(-2.0 * np.log(1.0 - level))) ** 2


def test_containment_radius_to_sigma_inverts_the_gaussian_containment():
    """A 90% radius is 2.1460 sigma, which is the number the synthesis hangs on."""
    sigma = gcn_skymaps.containment_radius_to_sigma(1.0, 0.9)
    assert sigma == pytest.approx(1.0 / 2.1460, rel=1e-4)
    # Round-tripping the relation is the property that actually matters.
    assert sigma * np.sqrt(-2.0 * np.log(1.0 - 0.9)) == pytest.approx(1.0)


@pytest.mark.parametrize("radius,containment", [(0.0, 0.9), (-1.0, 0.9), (1.0, 0.0), (1.0, 1.0)])
def test_containment_radius_to_sigma_rejects_impossible_inputs(radius, containment):
    """A zero radius or a containment of 1 has no finite Gaussian, so refuse rather than emit inf."""
    with pytest.raises(ValueError):
        gcn_skymaps.containment_radius_to_sigma(radius, containment)


def test_synthetic_map_order_resolves_finer_regions_with_finer_tiles():
    """An arcminute Einstein Probe circle needs a much finer order than a degree-scale one."""
    coarse = gcn_skymaps.synthetic_map_order(1.0)
    fine = gcn_skymaps.synthetic_map_order(0.01)
    assert fine > coarse
    assert gcn_skymaps.SYNTHETIC_MAP_MIN_ORDER <= coarse <= gcn_skymaps.SYNTHETIC_MAP_MAX_ORDER
    assert fine <= gcn_skymaps.SYNTHETIC_MAP_MAX_ORDER
    # A region far finer than order 13 clamps rather than exploding the tile count.
    assert gcn_skymaps.synthetic_map_order(1e-9) == gcn_skymaps.SYNTHETIC_MAP_MAX_ORDER


def test_refined_tiling_covers_the_whole_sky_exactly_once():
    """crossmatch reads an uncovered target as undefined, so the MOC has to be complete."""
    orders, indices = gcn_skymaps.refine_tiles(TEST_RA, TEST_DEC, np.deg2rad(1.0), max_order=6)
    areas = ah.nside_to_pixel_area(2**orders).to_value(u.sr)
    assert np.sum(areas) == pytest.approx(FULL_SKY_SR, rel=1e-12)
    # No tile may contain another, or the map is not a valid MOC.
    tiles = set(zip(orders.tolist(), indices.tolist(), strict=True))
    assert len(tiles) == len(orders)
    for order, index in tiles:
        ancestors = {
            (parent_order, index >> (2 * (order - parent_order)))
            for parent_order in range(min(orders), order)
        }
        assert not (ancestors & tiles)


def test_synthesized_map_is_normalized_and_reproduces_the_analytic_credible_areas():
    """This is the test that says the synthesized region means what the mission quoted."""
    table = gcn_skymaps.synthesize_moc_table(circle_localization())
    level, ipix = ah.uniq_to_level_ipix(table["UNIQ"])
    areas = ah.nside_to_pixel_area(ah.level_to_nside(level)).to_value(u.sr)
    assert np.sum(table["PROBDENSITY"] * areas) == pytest.approx(1.0, rel=1e-9)

    result = crossmatch(
        read_sky_map_from_table(table),
        SkyCoord([TEST_RA] * u.deg, [TEST_DEC] * u.deg),
        contours=(0.5, TEST_CONTAINMENT),
    )
    expected_50 = gaussian_credible_area_deg2(TEST_RADIUS_DEG, TEST_CONTAINMENT, 0.5)
    expected_90 = gaussian_credible_area_deg2(TEST_RADIUS_DEG, TEST_CONTAINMENT, TEST_CONTAINMENT)
    assert result.contour_areas[0] == pytest.approx(expected_50, rel=AREA_TOLERANCE)
    # The 90% area must come back as the circle the mission actually quoted.
    assert result.contour_areas[1] == pytest.approx(np.pi * TEST_RADIUS_DEG**2, rel=AREA_TOLERANCE)
    assert expected_90 == pytest.approx(np.pi * TEST_RADIUS_DEG**2, rel=1e-9)


def read_sky_map_from_table(table):
    """Round-trip a synthesized table through FITS, the way a consumer would read it.

    Parameters
    ----------
    table : astropy.table.Table
        Table from synthesize_moc_table().

    Returns
    -------
    astropy.table.Table
        The map as read back by ligo.skymap.
    """
    import io

    from ligo.skymap.io import write_sky_map

    buffer = io.BytesIO()
    write_sky_map(buffer, table, moc=True, nest=True)
    buffer.seek(0)
    return read_sky_map(buffer, moc=True)


def test_synthesized_map_peaks_at_the_quoted_position():
    """A map whose mode sits somewhere else would silently misdirect every crossmatch."""
    table = gcn_skymaps.synthesize_moc_table(circle_localization())
    level, ipix = ah.uniq_to_level_ipix(table["UNIQ"])
    peak = int(np.argmax(table["PROBDENSITY"]))
    lon, lat = ah.healpix_to_lonlat(ipix[peak], ah.level_to_nside(level[peak]), order="nested")
    separation = SkyCoord(TEST_RA * u.deg, TEST_DEC * u.deg).separation(SkyCoord(lon, lat))
    assert separation.deg < TEST_RADIUS_DEG / 4


def test_elliptical_region_is_elongated_along_its_position_angle():
    """ra_dec_error's third element is a position angle north through east; honour it."""
    localization = gcn_notices.Localization(
        ra=TEST_RA,
        dec=TEST_DEC,
        semi_major_deg=2.0,
        semi_minor_deg=0.5,
        position_angle_deg=0.0,
        containment_probability=TEST_CONTAINMENT,
    )
    table = gcn_skymaps.synthesize_moc_table(localization)
    sky_map = read_sky_map_from_table(table)
    center = SkyCoord(TEST_RA * u.deg, TEST_DEC * u.deg)
    # Position angle 0 is due north, so a point 1 deg north should stay far likelier than one
    # 1 deg east.
    along = center.directional_offset_by(0.0 * u.deg, 1.0 * u.deg)
    across = center.directional_offset_by(90.0 * u.deg, 1.0 * u.deg)
    result = crossmatch(sky_map, SkyCoord([along, across]))
    assert result.searched_prob[0] < result.searched_prob[1]


def test_disc_model_puts_all_probability_inside_the_quoted_radius():
    """The alternative model is a hard-edged disc: uniform inside, exactly zero outside."""
    table = gcn_skymaps.synthesize_moc_table(
        circle_localization(), model=gcn_skymaps.SYNTHETIC_MAP_MODEL_DISC
    )
    level, ipix = ah.uniq_to_level_ipix(table["UNIQ"])
    nside = ah.level_to_nside(level)
    areas = ah.nside_to_pixel_area(nside).to_value(u.sr)
    assert np.sum(table["PROBDENSITY"] * areas) == pytest.approx(1.0, rel=1e-9)

    lon, lat = ah.healpix_to_lonlat(ipix, nside, order="nested")
    separation = SkyCoord(TEST_RA * u.deg, TEST_DEC * u.deg).separation(SkyCoord(lon, lat)).deg
    inside = separation <= TEST_RADIUS_DEG
    density = np.asarray(table["PROBDENSITY"])
    # Nothing outside the quoted radius, and one single density inside it.
    assert np.all(density[~inside] == 0.0)
    assert np.all(density[inside] > 0.0)
    assert np.unique(density[inside]).size == 1
    # Unlike the Gaussian, the enclosed probability is the whole of it by the quoted radius.
    assert np.sum(density[inside] * areas[inside]) == pytest.approx(1.0, rel=1e-9)


def test_synthesize_rejects_a_localization_with_no_error_region():
    """BOOM's optical positions quote no error, and fabricating one is not an option."""
    localization = gcn_notices.Localization(ra=TEST_RA, dec=TEST_DEC)
    with pytest.raises(ValueError):
        gcn_skymaps.synthesize_moc_table(localization)


def test_synthesize_rejects_an_unknown_model():
    """A typo in the model constant should fail loudly, not fall through to a default."""
    with pytest.raises(ValueError):
        gcn_skymaps.synthesize_moc_table(circle_localization(), model="triangular")


def test_write_synthetic_skymap_is_readable_and_names_its_provenance(tmp_path):
    """A synthesized file must be readable by the same code that reads a real LVK map."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(3))
    path = tmp_path / "synthetic.multiorder.fits"
    gcn_skymaps.write_synthetic_skymap(path, record.localizations[0], record)
    sky_map = read_sky_map(str(path), moc=True)
    assert set(sky_map.colnames) == {"UNIQ", "PROBDENSITY"}
    # ligo.skymap normalizes a numeric OBJECT back to an int, so compare as text.
    assert str(sky_map.meta["objid"]) == record.event_id
    assert sky_map.meta["instruments"] == {"Swift/BAT-GUANO"}
    assert "synthetic" in sky_map.meta["creator"]


def test_resolve_prefers_the_inline_map_over_synthesizing(tmp_path):
    """GUANO record 2 ships a real map; re-deriving one from a radius would lose information."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(2))
    path, source = gcn_skymaps.resolve_skymap(
        tmp_path / "stem", record.localizations[0], record, fetch_remote=False
    )
    assert source == gcn_skymaps.SKYMAP_SOURCE_INLINE
    assert path.name.endswith(gcn_skymaps.REAL_SKYMAP_SUFFIX)
    assert path.read_bytes() == build_moc_fits_bytes()


def test_resolve_synthesizes_when_the_mission_supplied_no_map(tmp_path):
    """The Einstein Probe and preliminary IceCube case: a circle and nothing else."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(3))
    path, source = gcn_skymaps.resolve_skymap(
        tmp_path / "stem", record.localizations[0], record, fetch_remote=False
    )
    assert source == gcn_skymaps.SKYMAP_SOURCE_SYNTHESIZED
    # The file name has to advertise that this region is modelled, not observed.
    assert path.name.endswith(gcn_skymaps.SYNTHETIC_SKYMAP_SUFFIX)
    assert read_sky_map(str(path), moc=True) is not None


def test_resolve_returns_nothing_when_there_is_nothing_to_map(tmp_path):
    """GUANO record 1 has no localization at all; that is not an error."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(3))
    localization = gcn_notices.Localization(ra=TEST_RA, dec=TEST_DEC)
    assert gcn_skymaps.resolve_skymap(tmp_path / "stem", localization, record) == (None, None)
    assert list(tmp_path.iterdir()) == []


def test_resolve_downloads_a_hosted_map(tmp_path, monkeypatch):
    """IceCube's revised alerts publish the real map by URL, and it beats the circle."""
    record = gcn_notices.parse_notice(
        gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE, icecube_gold_bronze(record_number=1)
    )
    payload = build_moc_fits_bytes()
    seen = {}

    def fake_download(url, **kwargs):
        seen["url"] = url
        return payload

    monkeypatch.setattr(gcn_skymaps, "download_skymap", fake_download)
    path, source = gcn_skymaps.resolve_skymap(tmp_path / "stem", record.localizations[0], record)
    assert source == gcn_skymaps.SKYMAP_SOURCE_URL
    assert seen["url"] == record.localizations[0].healpix_url
    assert path.read_bytes() == payload


def test_resolve_falls_back_to_synthesis_when_the_download_fails(tmp_path, monkeypatch):
    """A network failure must not cost us the localization we can still model from the circle."""
    record = gcn_notices.parse_notice(
        gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE, icecube_gold_bronze(record_number=1)
    )

    def fail(url, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(gcn_skymaps, "download_skymap", fail)
    with pytest.warns(UserWarning, match="could not fetch skymap"):
        path, source = gcn_skymaps.resolve_skymap(tmp_path / "stem", record.localizations[0], record)
    assert source == gcn_skymaps.SKYMAP_SOURCE_SYNTHESIZED
    assert path.name.endswith(gcn_skymaps.SYNTHETIC_SKYMAP_SUFFIX)


def test_resolve_skips_the_download_when_remote_fetching_is_disabled(tmp_path, monkeypatch):
    """An offline run should synthesize rather than block on a URL it was told not to fetch."""
    record = gcn_notices.parse_notice(
        gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE, icecube_gold_bronze(record_number=1)
    )

    def fail(url, **kwargs):
        raise AssertionError("download attempted despite fetch_remote=False")

    monkeypatch.setattr(gcn_skymaps, "download_skymap", fail)
    _, source = gcn_skymaps.resolve_skymap(
        tmp_path / "stem", record.localizations[0], record, fetch_remote=False
    )
    assert source == gcn_skymaps.SKYMAP_SOURCE_SYNTHESIZED


def test_fallback_radius_is_off_by_default_and_applies_when_set():
    """Defaulting to "no map" keeps us from inventing a region the mission never claimed."""
    localization = gcn_notices.Localization(ra=TEST_RA, dec=TEST_DEC)
    assert gcn_skymaps.effective_localization(localization) is localization
    widened = gcn_skymaps.effective_localization(localization, fallback_radius_deg=0.25)
    assert widened.has_error_region
    assert widened.semi_major_deg == 0.25
    # A localization that already quotes an error keeps it.
    quoted = circle_localization()
    assert gcn_skymaps.effective_localization(quoted, fallback_radius_deg=99.0) is quoted


def test_download_skymap_refuses_an_oversized_response(monkeypatch):
    """A redirect to something huge should fail rather than fill the store's filesystem."""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"x" * (chunk_size or 1)

    monkeypatch.setattr(gcn_skymaps.requests, "get", lambda *a, **kw: FakeResponse())
    with pytest.raises(ValueError, match="exceeds"):
        gcn_skymaps.download_skymap("https://example.org/map.fits", max_bytes=10)


def test_einstein_probe_arcminute_circle_stays_tractable():
    """An arcminute region is the stress case a flat all-sky map could not represent at all."""
    localization = circle_localization(radius_deg=0.02)
    table = gcn_skymaps.synthesize_moc_table(localization)
    level, ipix = ah.uniq_to_level_ipix(table["UNIQ"])
    areas = ah.nside_to_pixel_area(ah.level_to_nside(level)).to_value(u.sr)
    assert np.sum(areas) == pytest.approx(FULL_SKY_SR, rel=1e-12)
    assert np.sum(table["PROBDENSITY"] * areas) == pytest.approx(1.0, rel=1e-9)
    # A flat map at this resolution would be hundreds of millions of pixels.
    assert len(table) < 100_000
    result = crossmatch(
        read_sky_map_from_table(table),
        SkyCoord([TEST_RA] * u.deg, [TEST_DEC] * u.deg),
        contours=(0.9,),
    )
    assert result.contour_areas[0] == pytest.approx(np.pi * 0.02**2, rel=AREA_TOLERANCE)
