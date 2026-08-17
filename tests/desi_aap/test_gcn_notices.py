"""Tests for parsing GCN notices into NoticeRecords."""

import json

import pytest
from gcn_examples import (
    BOOM_CROSSMATCH_ID,
    BOOM_TARGET_NAME,
    EINSTEIN_PROBE_TRIGGER_ID,
    GUANO_RA,
    GUANO_RA_DEC_ERROR,
    GUANO_TRIGGER_ID,
    ICECUBE_EVENT_NAME,
    ICECUBE_HEALPIX_URL,
    IGWN_SUPEREVENT_ID,
    LVK_NU_REFERENCE_ID,
    boom_alert,
    einstein_probe_wxt,
    icecube_gold_bronze,
    icecube_lvk_nu_track_search,
    igwn_gwalert,
    swift_bat_guano,
)

from desi_aap import gcn_notices


def test_every_default_topic_has_a_route_and_a_parser():
    """Nothing we subscribe to should fall through to the generic parser unnoticed."""
    assert set(gcn_notices.DEFAULT_TOPICS) == set(gcn_notices.TOPIC_ROUTING)
    assert set(gcn_notices.NOTICE_PARSERS) == set(gcn_notices.TOPIC_ROUTING)


def test_gw_alerts_route_apart_from_grb_and_neutrino_localizations():
    """The whole point of the layout is that GW maps do not share a tree with the rest."""
    categories = {topic: category for topic, (category, _) in gcn_notices.TOPIC_ROUTING.items()}
    assert categories[gcn_notices.TOPIC_IGWN_GWALERT] == gcn_notices.CATEGORY_GW
    assert categories[gcn_notices.TOPIC_SWIFT_BAT_GUANO] == gcn_notices.CATEGORY_GRB
    assert categories[gcn_notices.TOPIC_EINSTEIN_PROBE_WXT] == gcn_notices.CATEGORY_GRB
    assert categories[gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE] == gcn_notices.CATEGORY_NEUTRINO
    assert categories[gcn_notices.TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH] == gcn_notices.CATEGORY_NEUTRINO


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, (None, None, None)),
        (0.5, (0.5, 0.5, 0.0)),
        ([1.5], (1.5, 1.5, 0.0)),
        ([1.5, 0.5], (1.5, 0.5, 0.0)),
        ([1.5, 0.5, 30.0], (1.5, 0.5, 30.0)),
        ([], (None, None, None)),
        ("nonsense", (None, None, None)),
    ],
)
def test_parse_ra_dec_error_handles_both_polymorphic_forms(value, expected):
    """core/Localization allows a scalar circle or a 1-3 element ellipse; both must parse."""
    assert gcn_notices.parse_ra_dec_error(value) == expected


@pytest.mark.parametrize("value,expected", [("abc", ("abc",)), (["a", "b"], ("a", "b")), (None, ())])
def test_as_id_tuple_accepts_string_or_array(value, expected):
    """core/Event declares event_name and id as "string or array of strings"."""
    assert gcn_notices.as_id_tuple(value) == expected


def test_parse_notice_accepts_bytes_str_and_dict():
    """Kafka hands us bytes; tests and replays hand us text or a parsed payload."""
    payload = einstein_probe_wxt()
    text = json.dumps(payload)
    records = [
        gcn_notices.parse_notice(gcn_notices.TOPIC_EINSTEIN_PROBE_WXT, form)
        for form in (payload, text, text.encode("utf-8"))
    ]
    assert {record.event_id for record in records} == {EINSTEIN_PROBE_TRIGGER_ID}


@pytest.mark.parametrize("body", [b"not json", b"[1, 2, 3]", b'"a string"'])
def test_parse_notice_rejects_non_object_bodies(body):
    """A body that is not a JSON object is a quarantine case, not something to guess at."""
    with pytest.raises(ValueError):
        gcn_notices.parse_notice(gcn_notices.TOPIC_EINSTEIN_PROBE_WXT, body)


def test_igwn_gwalert_parses_superevent_and_inline_skymap():
    """IGWN localization arrives only as a base64 multi-order FITS map under event.skymap."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_IGWN_GWALERT, igwn_gwalert())
    assert record.category == gcn_notices.CATEGORY_GW
    assert record.event_id == IGWN_SUPEREVENT_ID
    assert record.alert_type == "preliminary"
    # IGWN publishes no sequence number at all, so ordering has to lean on time_created.
    assert record.record_number is None
    assert record.notice_time == "2018-11-01T22:34:49Z"
    assert record.trigger_time == "2018-11-01T22:22:46.654Z"
    assert not record.is_retraction
    assert len(record.localizations) == 1
    localization = record.localizations[0]
    assert localization.label is None
    assert localization.healpix_bytes.startswith(b"SIMPLE")
    # There is no RA/Dec on this topic, so nothing should be synthesized from it.
    assert not localization.has_point
    assert not localization.has_error_region
    assert localization.has_real_map


def test_igwn_gwalert_keeps_the_combined_external_skymap_separately():
    """external_coinc's combined map is a second localization and needs its own file name."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_IGWN_GWALERT, igwn_gwalert(with_external_coinc=True))
    labels = [localization.label for localization in record.localizations]
    assert labels == [None, gcn_notices.COMBINED_SKYMAP_LABEL]
    assert "12345" in record.related_ids


def test_igwn_retraction_has_no_localization():
    """A retraction nulls event and external_coinc, so there is nothing left to localize."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_IGWN_GWALERT, igwn_gwalert(alert_type="RETRACTION"))
    assert record.is_retraction
    assert record.alert_type == gcn_notices.ALERT_TYPE_RETRACTION
    assert record.localizations == ()


@pytest.mark.parametrize(
    "raw,expected",
    [("PRELIMINARY", "preliminary"), ("EARLYWARNING", "earlywarning"), ("update", "update")],
)
def test_alert_type_case_is_normalized_across_schema_families(raw, expected):
    """IGWN spells alert types upper case and gcn.notices.* lower case; ranks must compare."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_IGWN_GWALERT, igwn_gwalert(alert_type=raw))
    assert record.alert_type == expected
    assert record.alert_type_rank == gcn_notices.ALERT_TYPE_RANK[expected]


def test_einstein_probe_defaults_containment_to_the_schema_default():
    """WXT omits containment_probability, which core/Localization defines as 0.9."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_EINSTEIN_PROBE_WXT, einstein_probe_wxt())
    assert record.category == gcn_notices.CATEGORY_GRB
    assert record.event_id == EINSTEIN_PROBE_TRIGGER_ID
    # WXT does not compose core/Alert at all, so it has neither alert type nor notice time.
    assert record.alert_type is None
    assert record.notice_time is None
    assert record.trigger_time == "2024-02-26T05:31:26.42Z"
    localization = record.localizations[0]
    assert localization.containment_probability == gcn_notices.DEFAULT_CONTAINMENT_PROBABILITY
    assert localization.systematic_included is gcn_notices.DEFAULT_SYSTEMATIC_INCLUDED
    assert localization.semi_major_deg == localization.semi_minor_deg == 0.02
    assert localization.has_error_region
    assert not localization.has_real_map


def test_guano_initial_record_carries_no_localization():
    """GUANO's first notice is a trigger report; the localization arrives in later records."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(1))
    assert record.event_id == GUANO_TRIGGER_ID
    assert record.record_number == 1
    assert record.alert_type == "initial"
    assert record.localizations == ()
    # follow_up_event is the external trigger GUANO searched around: the join back to it.
    assert "Fermi 694215970" in record.related_ids


def test_guano_map_record_prefers_the_inline_healpix_file():
    """Record 2 attaches a real map, which must be kept rather than re-derived."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(2))
    localization = record.localizations[0]
    assert localization.healpix_bytes.startswith(b"SIMPLE")
    assert localization.has_real_map


def test_guano_circle_record_parses_the_quoted_error_region():
    """Record 3 quotes an error circle with systematics folded in."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(3))
    localization = record.localizations[0]
    assert (localization.ra, localization.dec) == (GUANO_RA, pytest.approx(25.139))
    assert localization.semi_major_deg == GUANO_RA_DEC_ERROR
    assert localization.systematic_included is True
    assert localization.has_error_region
    assert not localization.has_real_map


def test_guano_retraction_is_detected_from_lower_case_alert_type():
    """gcn.notices.* retractions are ordinary populated messages, unlike IGWN's null payload."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(4))
    assert record.is_retraction
    assert record.record_number == 4
    assert record.localizations == ()


def test_icecube_gold_bronze_revision_exposes_the_hosted_map():
    """IceCube warns the circularized error is approximate, so the revised map wins."""
    record = gcn_notices.parse_notice(
        gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE, icecube_gold_bronze(record_number=1)
    )
    assert record.category == gcn_notices.CATEGORY_NEUTRINO
    assert record.event_id == ICECUBE_EVENT_NAME
    assert record.record_number == 1
    localization = record.localizations[0]
    assert localization.healpix_url == ICECUBE_HEALPIX_URL
    assert localization.has_real_map
    # The circle is still parsed, so it can serve as a fallback if the fetch fails.
    assert localization.has_error_region


def test_icecube_gold_bronze_preliminary_has_only_the_circle():
    """Record 0 publishes no map, which is exactly when synthesis has to step in."""
    record = gcn_notices.parse_notice(
        gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE, icecube_gold_bronze(record_number=0)
    )
    assert record.record_number == 0
    localization = record.localizations[0]
    assert localization.healpix_url is None
    assert not localization.has_real_map
    assert localization.has_error_region
    assert localization.systematic_included is False


@pytest.mark.parametrize("nested", [True, False])
def test_icecube_lvk_search_accepts_nested_and_flat_localizations(nested):
    """The schema puts these fields flat and the published example nests them; accept both."""
    record = gcn_notices.parse_notice(
        gcn_notices.TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH,
        icecube_lvk_nu_track_search(nested_localization=nested),
    )
    # This topic has no id of its own: it is keyed by the superevent it followed up.
    assert record.event_id == LVK_NU_REFERENCE_ID
    assert LVK_NU_REFERENCE_ID in record.related_ids
    assert len(record.localizations) == 1
    localization = record.localizations[0]
    assert localization.label == "138590_39138551"
    assert localization.ra == pytest.approx(17.48)
    assert localization.dec == pytest.approx(16.15)
    assert localization.semi_major_deg == pytest.approx(0.5)


def test_icecube_lvk_search_labels_each_coincident_neutrino():
    """One notice can report several neutrinos, and each needs its own map file name."""
    payload = icecube_lvk_nu_track_search()
    second = json.loads(json.dumps(payload["coincident_events"][0]))
    second["id"] = ["138590_99999999"]
    payload["coincident_events"].append(second)
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH, payload)
    assert [localization.label for localization in record.localizations] == [
        "138590_39138551",
        "138590_99999999",
    ]


def test_boom_notice_keys_on_its_target_and_records_the_crossmatch():
    """BOOM has no top-level id; its value is the crossmatch back to a GW or GRB event."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_BOOM, boom_alert())
    assert record.category == gcn_notices.CATEGORY_OPTICAL
    assert record.event_id == BOOM_TARGET_NAME
    assert record.related_ids == (BOOM_CROSSMATCH_ID,)
    assert record.trigger_time == "2025-11-18T05:00:00.00Z"
    localization = record.localizations[0]
    assert localization.label == BOOM_TARGET_NAME
    assert localization.has_point
    # An arcsecond-scale optical position quotes no error region, and we do not invent one.
    assert not localization.has_error_region


def test_unknown_topic_falls_back_to_the_core_shape():
    """A newly added gcn.notices.* mission should still store rather than raise."""
    record = gcn_notices.parse_notice("gcn.notices.newmission.alert", swift_bat_guano(3))
    assert record.event_id == GUANO_TRIGGER_ID
    assert record.localizations[0].has_error_region


def test_notice_without_identifier_still_gets_a_stable_event_id():
    """A payload we cannot identify is worth keeping; it must not collide with a real event."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, {"ra": 1.0, "dec": 2.0})
    assert record.event_id == gcn_notices.UNKNOWN_EVENT_ID


def test_summary_is_json_serializable_and_omits_map_bytes():
    """Summaries go into the index, so they must not carry megabytes of FITS."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_IGWN_GWALERT, igwn_gwalert())
    json.dumps(record.summary())
    summary = record.localizations[0].summary()
    json.dumps(summary)
    assert summary["has_inline_map"] is True
    assert "healpix_bytes" not in summary


def test_undecodable_healpix_file_is_dropped_rather_than_stored():
    """Truncated base64 must not be written out as a corrupt FITS file."""
    assert gcn_notices.decode_healpix_file("not valid base64!!") is None
    assert gcn_notices.decode_healpix_file(None) is None
    assert gcn_notices.decode_healpix_file("") is None


def test_notice_instruments_reports_mission_and_instrument():
    """The synthesized map's FITS header names whoever reported the event."""
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, swift_bat_guano(3))
    assert gcn_notices.notice_instruments(record) == ["Swift/BAT-GUANO"]
    igwn = gcn_notices.parse_notice(gcn_notices.TOPIC_IGWN_GWALERT, igwn_gwalert())
    assert gcn_notices.notice_instruments(igwn) == []
