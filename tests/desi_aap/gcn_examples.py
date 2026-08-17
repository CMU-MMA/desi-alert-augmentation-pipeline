"""Example GCN notice payloads, transcribed from the published schema examples.

Field names and value shapes follow gcn-schema's ``*.example.json`` files and
igwn-gwalert-schema's published example, because the point of these fixtures is to pin the
parser to the real wire format rather than to our idea of it. Numeric values are abridged
where their exact magnitude does not matter to the parser.

Skymap bytes are generated rather than transcribed: a real LVK map is megabytes of base64, so
build_moc_fits_bytes() writes a small but genuinely valid multi-order FITS file instead, which
also exercises the base64 round-trip.
"""

import base64
import io

import numpy as np
from astropy.table import Table
from ligo.skymap import moc
from ligo.skymap.io import write_sky_map

# Order of the throwaway map used for inline-skymap fixtures. Order 1 is 48 tiles: valid,
# all-sky, and small enough to embed in a test payload.
FIXTURE_MOC_ORDER = 1

# Swift GUANO's published examples run record 1 (trigger only) through record 4 (retraction),
# which is the lifecycle the store's version ordering has to get right.
GUANO_TRIGGER_ID = "694215995"
GUANO_RA = 336.26
GUANO_DEC = 25.139
GUANO_RA_DEC_ERROR = 0.5

# IceCube gold/bronze: record 0 is preliminary with no map, record 1 revises it and adds one.
ICECUBE_EVENT_NAME = "IceCube-260425A"
ICECUBE_EVENT_ID = "137840_57034692"
ICECUBE_HEALPIX_URL = (
    "https://roc-2.icecube.wisc.edu/public/alerts/IceCube-260425A_skymap_probdensity_multiorder.fits.gz"
)

EINSTEIN_PROBE_TRIGGER_ID = "01708973486"
IGWN_SUPEREVENT_ID = "MS181101ab"
LVK_NU_REFERENCE_ID = "S230914ak"
BOOM_TARGET_NAME = "ZTF25acffkdr"
BOOM_CROSSMATCH_ID = "S251117bs"


def build_moc_fits_bytes(order=FIXTURE_MOC_ORDER):
    """Build a small, valid NUNIQ multi-order FITS map as raw bytes.

    Parameters
    ----------
    order : int, optional
        HEALPix order of the uniform map.

    Returns
    -------
    bytes
        FITS file contents, readable by ligo.skymap.io.read_sky_map.
    """
    npix = 12 * 4**order
    uniq = moc.nest2uniq(np.int8(order), np.arange(npix))
    density = np.full(npix, 1.0 / (4.0 * np.pi))
    buffer = io.BytesIO()
    write_sky_map(buffer, Table({"UNIQ": uniq, "PROBDENSITY": density}), moc=True, nest=True)
    return buffer.getvalue()


def build_moc_fits_base64(order=FIXTURE_MOC_ORDER):
    """Return build_moc_fits_bytes() encoded the way a JSON notice carries it.

    Parameters
    ----------
    order : int, optional
        HEALPix order of the uniform map.

    Returns
    -------
    str
        Base64 text.
    """
    return base64.b64encode(build_moc_fits_bytes(order=order)).decode("ascii")


def igwn_gwalert(alert_type="PRELIMINARY", with_skymap=True, with_external_coinc=False):
    """Build an ``igwn.gwalert`` payload.

    Parameters
    ----------
    alert_type : str, optional
        One of EARLYWARNING, PRELIMINARY, INITIAL, UPDATE, RETRACTION.
    with_skymap : bool, optional
        Whether ``event.skymap`` carries an inline map.
    with_external_coinc : bool, optional
        Whether to add an ``external_coinc`` block with its combined skymap.

    Returns
    -------
    dict
        The payload.
    """
    if alert_type == "RETRACTION":
        # A retraction nulls both event and external_coinc, so there is nothing to localize.
        return {
            "alert_type": "RETRACTION",
            "time_created": "2018-11-01T23:45:00Z",
            "superevent_id": IGWN_SUPEREVENT_ID,
            "urls": {"gracedb": f"https://example.org/superevents/{IGWN_SUPEREVENT_ID}/view/"},
            "event": None,
            "external_coinc": None,
        }

    event = {
        "time": "2018-11-01T22:22:46.654Z",
        "far": 9.11069936486e-14,
        "significant": True,
        "instruments": ["H1", "L1", "V1"],
        "group": "CBC",
        "pipeline": "gstlal",
        "search": "MDC",
        "properties": {"HasNS": 0.95, "HasRemnant": 0.91, "HasMassGap": 0.01},
        "classification": {"BNS": 0.95, "NSBH": 0.01, "BBH": 0.03, "Terrestrial": 0.01},
        "duration": None,
        "central_frequency": None,
        "skymap_filename": "bayestar.multiorder.fits",
        "skymap": build_moc_fits_base64() if with_skymap else None,
    }
    external = None
    if with_external_coinc:
        external = {
            "gcn_notice_id": "12345",
            "ivorn": "ivo://nasa.gsfc.gcn/Fermi#GBM_Fin_Pos2018-11-01T22:22:46.65",
            "observatory": "Fermi",
            "search": "GRB",
            "time_difference": -3.1,
            "time_coincidence_far": 1.9e-09,
            "time_sky_position_coincidence_far": 7.4e-11,
            "combined_skymap_filename": "combined-ext.multiorder.fits",
            "combined_skymap": build_moc_fits_base64(),
        }
    return {
        "alert_type": alert_type,
        "time_created": "2018-11-01T22:34:49Z",
        "superevent_id": IGWN_SUPEREVENT_ID,
        "urls": {"gracedb": f"https://example.org/superevents/{IGWN_SUPEREVENT_ID}/view/"},
        "event": event,
        "external_coinc": external,
    }


def einstein_probe_wxt():
    """Build a ``gcn.notices.einstein_probe.wxt.alert`` payload.

    This topic does not compose core/Alert.schema.json, so it deliberately has no
    ``alert_type``, ``alert_tense`` or ``alert_datetime``, and it omits
    ``containment_probability`` so the schema default of 0.9 applies.

    Returns
    -------
    dict
        The payload.
    """
    return {
        "instrument": "WXT",
        "trigger_time": "2024-02-26T05:31:26.42Z",
        "id": [EINSTEIN_PROBE_TRIGGER_ID],
        "ra": 227.4437,
        "dec": -13.7053,
        "ra_dec_error": 0.02,
        "image_energy_range": [0.5, 4],
        "net_count_rate": 0.1,
        "image_snr": 12.7,
        "additional_info": "The net count rate is from an image accumulated over up to 20 min.",
    }


def icecube_gold_bronze(record_number=1, with_healpix_url=True):
    """Build a ``gcn.notices.icecube.gold_bronze_track_alerts`` payload.

    Parameters
    ----------
    record_number : int, optional
        0 for the preliminary notice, which quotes no map, or 1 for the revision, which adds
        one by URL and folds systematics into the quoted error.
    with_healpix_url : bool, optional
        Whether ``healpix_url`` is populated.

    Returns
    -------
    dict
        The payload.
    """
    return {
        "mission": "IceCube",
        "instrument": "IC86",
        "messenger": "Neutrino",
        "record_number": record_number,
        "alert_tense": "current",
        "alert_type": "update" if record_number else "initial",
        "alert_datetime": "2026-04-25T06:25:00.00Z",
        "trigger_time": "2026-04-25T06:22:15.13Z",
        "event_name": [ICECUBE_EVENT_NAME],
        "id": [ICECUBE_EVENT_ID],
        "pipeline": "Gold Track Alert",
        "alert_topology": "Track",
        "number_of_events": 1,
        "ra": 347.08,
        "dec": 19.19,
        "ra_dec_error": 0.6,
        "containment_probability": 0.9,
        "systematic_included": bool(record_number),
        "healpix_url": ICECUBE_HEALPIX_URL if (record_number and with_healpix_url) else None,
        "far": 8.029e-8,
        "p_astro": 0.34064,
        "nu_energy": 127.29,
    }


def icecube_lvk_nu_track_search(nested_localization=True):
    """Build a ``gcn.notices.icecube.lvk_nu_track_search`` payload.

    Parameters
    ----------
    nested_localization : bool, optional
        Whether each coincident event nests its localization under a ``localization`` key, as
        the published example does, or carries the fields flat, as the schema and the IceCube
        mission docs describe. The parser has to accept both.

    Returns
    -------
    dict
        The payload.
    """
    localization = {
        "ra": 17.48,
        "dec": 16.15,
        "ra_dec_error": 0.5,
        "containment_probability": 0.9,
        "systematic_included": False,
    }
    event = {
        "event_dt": 12.91,
        "id": ["138590_39138551"],
        "event_pval_generic": 0.05,
        "event_pval_bayesian": None,
    }
    event.update({"localization": localization} if nested_localization else localization)
    return {
        "mission": "IceCube",
        "instrument": "IC86",
        "messenger": "Neutrino",
        "ref_ID": LVK_NU_REFERENCE_ID,
        "ref_type": "GW",
        "ref_instrument": "LVK",
        "reference": {"gcn.notices.LVK.alert": f"{LVK_NU_REFERENCE_ID}-2-Preliminary"},
        "trigger_time": "2023-09-14T09:01:12.00Z",
        "observation_start": "2023-09-14T08:52:52.00Z",
        "observation_stop": "2023-09-14T09:09:32.00Z",
        "observation_livetime": 1000,
        "pval_generic": 0.23,
        "pval_bayesian": None,
        "n_events_coincident": 1,
        "coincident_events": [event],
        "most_probable_direction": {"ra": 17.0, "dec": 16.0},
        "neutrino_flux_sensitivity_range": {
            "flux_sensitivity": [0.014, 0.13],
            "sensitive_energy_range": [280, 6100000],
        },
    }


def swift_bat_guano(record_number=3):
    """Build a ``gcn.notices.swift.bat.guano`` payload.

    Parameters
    ----------
    record_number : int, optional
        1 for the initial notice, which carries no localization at all; 2 for the revision
        that attaches an inline HEALPix map; 3 for the one that quotes an error circle
        instead; 4 for the retraction.

    Returns
    -------
    dict
        The payload.
    """
    payload = {
        "mission": "Swift",
        "instrument": "BAT-GUANO",
        "messenger": "EM",
        "record_number": record_number,
        "alert_tense": "current",
        "alert_type": "retraction" if record_number == 4 else ("initial" if record_number == 1 else "update"),
        "alert_datetime": f"2022-01-30T07:0{record_number}:00.00Z",
        "trigger_time": "2022-01-30T06:59:55.00Z",
        "id": [GUANO_TRIGGER_ID],
        "data_archive_page": f"https://guano.swift.psu.edu/trigger_report?id={GUANO_TRIGGER_ID}",
        "follow_up_event": "Fermi 694215970",
        "follow_up_type": "GW",
    }
    if record_number != 4:
        # A retraction drops the Statistics block along with the localization.
        payload.update(
            {
                "rate_snr": 7.2,
                "rate_duration": 1.0,
                "rate_energy_range": [15, 350],
                "image_snr": 6.9,
                "image_duration": 1.0,
                "image_energy_range": [15, 350],
                "classification": {"GRB": 1},
                "far": 1.0e-7,
            }
        )
    if record_number == 2:
        payload.update({"healpix_file": build_moc_fits_base64(), "systematic_included": True})
    if record_number == 3:
        payload.update(
            {
                "ra": GUANO_RA,
                "dec": GUANO_DEC,
                "ra_dec_error": GUANO_RA_DEC_ERROR,
                "containment_probability": 0.9,
                "systematic_included": True,
            }
        )
    return payload


def boom_alert():
    """Build a ``gcn.notices.boom.alert`` payload.

    BOOM is a ZTF/Rubin broker, so the notice is a container of optical targets and
    photometry with no top-level event id or trigger time, and each target names the GCN event
    it was crossmatched to.

    Returns
    -------
    dict
        The payload.
    """
    return {
        "mission": "Boom",
        "instrument": "ZTF",
        "messenger": "EM",
        "record_number": 1,
        "alert_tense": "current",
        "alert_type": "initial",
        "alert_datetime": "2025-11-18T06:00:00.00Z",
        "data": {
            "targets": [
                {
                    "event_name": BOOM_TARGET_NAME,
                    "ra": 123.456,
                    "dec": 12.345,
                    "classification_scores": {"Type II": 0.9},
                    "gcn_crossmatch": [
                        {"ref_type": "GW", "ref_instrument": "LVK", "ref_ID": BOOM_CROSSMATCH_ID}
                    ],
                }
            ],
            "photometry": [
                {
                    "telescope": "ZTF",
                    "instrument": "ZTF",
                    "event_name": BOOM_TARGET_NAME,
                    "filter": "ztfg",
                    "mag": 19.2,
                    "mag_error": 0.1,
                    "mag_system": "AB",
                    "limiting_mag": 20.5,
                    "observation_start": "2025-11-18T05:00:00.00Z",
                }
            ],
        },
    }
