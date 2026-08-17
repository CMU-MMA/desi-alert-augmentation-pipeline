"""Parse GCN Kafka notices into one normalized record shape.

GCN carries two unrelated schema families on the topics we subscribe to, and they disagree
on nearly every convention:

* ``igwn.gwalert`` follows ``igwn-gwalert-schema``. It identifies events by
  ``superevent_id``, spells ``alert_type`` in upper case, has no sequence number, and
  publishes localization *only* as a base64 multi-order FITS skymap.
* Every ``gcn.notices.*`` topic composes NASA's ``gcn-schema`` ``core/*`` files. Those
  identify events by ``event_name``/``id``, spell ``alert_type`` in lower case, carry a
  ``record_number`` sequence, and publish localization as ``ra``/``dec``/``ra_dec_error``
  with an optional HEALPix file or URL.

parse_notice() flattens both into a NoticeRecord so the store and the skymap writer never
have to care which topic a notice arrived on.
"""

import base64
import binascii
import json
from dataclasses import dataclass, field

# Kafka topic names, verified against the JsonNoticeTypes list that drives gcn.nasa.gov's
# own subscription UI. Note that igwn.gwalert has no "gcn." prefix, and that the schema
# file names do not match the topic names (the gold/bronze schema is
# icecube/single_neutrino_alerts.schema.json), so topics cannot be derived from schema paths.
TOPIC_IGWN_GWALERT = "igwn.gwalert"
TOPIC_EINSTEIN_PROBE_WXT = "gcn.notices.einstein_probe.wxt.alert"
TOPIC_ICECUBE_GOLD_BRONZE = "gcn.notices.icecube.gold_bronze_track_alerts"
TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH = "gcn.notices.icecube.lvk_nu_track_search"
TOPIC_SWIFT_BAT_GUANO = "gcn.notices.swift.bat.guano"
TOPIC_BOOM = "gcn.notices.boom.alert"

# Store categories. GW maps land under CATEGORY_GW, and the GRB/neutrino localizations go
# in their own trees so they never mix with the gravitational-wave maps.
CATEGORY_GW = "gw"
CATEGORY_GRB = "grb"
CATEGORY_NEUTRINO = "neutrino"
CATEGORY_OPTICAL = "optical"

# Topic -> (store category, source directory name). The source name is what distinguishes
# two missions inside one category, e.g. grb/swift_bat_guano vs grb/einstein_probe_wxt.
TOPIC_ROUTING = {
    TOPIC_IGWN_GWALERT: (CATEGORY_GW, "lvk"),
    TOPIC_EINSTEIN_PROBE_WXT: (CATEGORY_GRB, "einstein_probe_wxt"),
    TOPIC_SWIFT_BAT_GUANO: (CATEGORY_GRB, "swift_bat_guano"),
    TOPIC_ICECUBE_GOLD_BRONZE: (CATEGORY_NEUTRINO, "icecube_gold_bronze"),
    TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH: (CATEGORY_NEUTRINO, "icecube_lvk_nu_track_search"),
    TOPIC_BOOM: (CATEGORY_OPTICAL, "boom"),
}

# Subscribed by default, in the order they are listed above.
DEFAULT_TOPICS = tuple(TOPIC_ROUTING)

# Defaults mandated by gcn-schema core/Localization.schema.json for absent keys. These are
# not our invention: a notice that omits containment_probability means 0.9, and one that
# omits systematic_included means the quoted error excludes systematics.
DEFAULT_CONTAINMENT_PROBABILITY = 0.9
DEFAULT_SYSTEMATIC_INCLUDED = False

# Normalized alert_type values, lower-cased from either schema family. IGWN uses
# EARLYWARNING/PRELIMINARY/INITIAL/UPDATE/RETRACTION; gcn.notices.* uses
# initial/subsequent/update/retraction. Ranks break ties when ordering notices for the
# "latest" pointer and a record_number is unavailable (IGWN has none).
ALERT_TYPE_RETRACTION = "retraction"
ALERT_TYPE_RANK = {
    "earlywarning": 0,
    "preliminary": 1,
    "initial": 2,
    "subsequent": 3,
    "update": 4,
    ALERT_TYPE_RETRACTION: 5,
}
UNKNOWN_ALERT_TYPE_RANK = -1

# Label given to the joint GW + external-trigger skymap that IGWN ships alongside the main
# one in external_coinc, so the two maps get distinct file names in the store.
COMBINED_SKYMAP_LABEL = "combined"

# Used when a notice carries no usable identifier at all, so the payload is still stored
# rather than dropped.
UNKNOWN_EVENT_ID = "unidentified"


def as_float(value):
    """Coerce a JSON value to float, returning None for anything unusable.

    Parameters
    ----------
    value : object
        Value from a parsed notice, possibly None or a non-numeric string.

    Returns
    -------
    float or None
        The value as a float, or None if it is missing or not convertible.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_id_tuple(value):
    """Normalize a gcn-schema identifier field to a tuple of strings.

    core/Event.schema.json declares both ``event_name`` and ``id`` as "string or array of
    strings", so every consumer has to handle both spellings.

    Parameters
    ----------
    value : str or list or None
        Raw identifier field.

    Returns
    -------
    tuple of str
        Identifiers with empty entries dropped, in their original order.
    """
    if value is None:
        return ()
    values = value if isinstance(value, list | tuple) else [value]
    return tuple(str(item) for item in values if item is not None and str(item) != "")


def decode_healpix_file(value):
    """Decode a base64 ``healpix_file``/``skymap`` field to raw FITS bytes.

    Parameters
    ----------
    value : str or bytes or None
        Base64 text as it appears in a JSON notice, or raw bytes as it appears in Avro.

    Returns
    -------
    bytes or None
        Decoded FITS bytes, or None if the field is absent or not decodable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bytes):
        return value or None
    try:
        return base64.b64decode(value, validate=True) or None
    except (binascii.Error, TypeError, ValueError):
        return None


def parse_ra_dec_error(value):
    """Split the polymorphic ``ra_dec_error`` field into semi-axes and a position angle.

    core/Localization.schema.json allows either a number, meaning a circle of that radius,
    or an array of one to three numbers, meaning ``[semi_major, semi_minor, position_angle]``
    with semi_minor defaulting to semi_major and the angle defaulting to zero. All values
    are degrees; the position angle runs north through east.

    Parameters
    ----------
    value : float or list or None
        Raw ``ra_dec_error`` field.

    Returns
    -------
    tuple
        ``(semi_major_deg, semi_minor_deg, position_angle_deg)``, each None if the field is
        missing or unusable.
    """
    if value is None:
        return (None, None, None)
    if isinstance(value, list | tuple):
        parts = [as_float(item) for item in value]
        parts = [part for part in parts if part is not None]
        if not parts:
            return (None, None, None)
        semi_major = parts[0]
        semi_minor = parts[1] if len(parts) > 1 else semi_major
        position_angle = parts[2] if len(parts) > 2 else 0.0
        return (semi_major, semi_minor, position_angle)
    semi_major = as_float(value)
    if semi_major is None:
        return (None, None, None)
    return (semi_major, semi_major, 0.0)


@dataclass(frozen=True)
class Localization:
    """One sky localization carried by a notice.

    A notice can carry more than one: IceCube's LVK track search reports a region per
    coincident neutrino, BOOM reports one per optical target, and IGWN adds a combined
    GW-plus-external map. ``label`` keeps them apart in file names; it is None for the
    notice's single or primary region.

    Attributes
    ----------
    label : str or None
        Sub-event key, e.g. a neutrino ``RUNID_EVENTID`` or an optical target name.
    ra, dec : float or None
        ICRS position in degrees.
    semi_major_deg, semi_minor_deg : float or None
        Error-region semi-axes in degrees; equal for a circular region.
    position_angle_deg : float or None
        Orientation of the semi-major axis in degrees, north through east.
    containment_probability : float
        Probability enclosed by the quoted error region.
    systematic_included : bool
        Whether the quoted error already includes systematics.
    healpix_bytes : bytes or None
        Real HEALPix FITS map that came inline with the notice.
    healpix_url : str or None
        URL of a real HEALPix map hosted by the mission.
    """

    label: str | None = None
    ra: float | None = None
    dec: float | None = None
    semi_major_deg: float | None = None
    semi_minor_deg: float | None = None
    position_angle_deg: float | None = None
    containment_probability: float = DEFAULT_CONTAINMENT_PROBABILITY
    systematic_included: bool = DEFAULT_SYSTEMATIC_INCLUDED
    healpix_bytes: bytes | None = None
    healpix_url: str | None = None

    @property
    def has_point(self):
        """Whether the localization has a usable RA/Dec centre."""
        return self.ra is not None and self.dec is not None

    @property
    def has_error_region(self):
        """Whether the localization has a positive-size error region to synthesize from."""
        return (
            self.has_point
            and self.semi_major_deg is not None
            and self.semi_minor_deg is not None
            and self.semi_major_deg > 0
            and self.semi_minor_deg > 0
        )

    @property
    def has_real_map(self):
        """Whether the mission supplied a real HEALPix map, inline or by URL."""
        return self.healpix_bytes is not None or bool(self.healpix_url)

    def summary(self):
        """Return a JSON-serializable summary, omitting any inline map bytes.

        Returns
        -------
        dict
            Localization fields suitable for the store index.
        """
        return {
            "label": self.label,
            "ra": self.ra,
            "dec": self.dec,
            "semi_major_deg": self.semi_major_deg,
            "semi_minor_deg": self.semi_minor_deg,
            "position_angle_deg": self.position_angle_deg,
            "containment_probability": self.containment_probability,
            "systematic_included": self.systematic_included,
            "healpix_url": self.healpix_url,
            "has_inline_map": self.healpix_bytes is not None,
        }


@dataclass(frozen=True)
class NoticeRecord:
    """A GCN notice normalized across both schema families.

    Attributes
    ----------
    topic : str
        Kafka topic the notice arrived on.
    category, source : str
        Store category ("gw", "grb", "neutrino", "optical") and mission directory name.
    event_id : str
        Identifier the store groups versions of this event under.
    event_names : tuple of str
        Every identifier the notice carried, primary first.
    related_ids : tuple of str
        Identifiers of other events this notice points at: the followed-up superevent for
        IceCube's LVK search, the external trigger for Swift GUANO, the crossmatched GCN
        event for BOOM. These are what let the pipeline join a counterpart back to its GW
        event.
    alert_type : str or None
        Lower-cased alert type, or None on topics that publish none (Einstein Probe WXT
        does not compose core/Alert.schema.json at all).
    record_number : int or None
        Mission sequence number. Always None for IGWN, which has no such field.
    trigger_time, notice_time : str or None
        Event/trigger time and notice-creation time, as the ISO-8601 strings from the wire.
    is_retraction : bool
        Whether the notice withdraws the event.
    localizations : tuple of Localization
        Sky localizations carried by the notice; empty if it carried none.
    payload : dict
        The parsed notice, stored verbatim.
    """

    topic: str
    category: str
    source: str
    event_id: str
    event_names: tuple = ()
    related_ids: tuple = ()
    alert_type: str | None = None
    record_number: int | None = None
    trigger_time: str | None = None
    notice_time: str | None = None
    is_retraction: bool = False
    localizations: tuple = ()
    payload: dict = field(default_factory=dict)

    @property
    def alert_type_rank(self):
        """Rank of this notice's alert type, for ordering versions of one event."""
        if self.alert_type is None:
            return UNKNOWN_ALERT_TYPE_RANK
        return ALERT_TYPE_RANK.get(self.alert_type, UNKNOWN_ALERT_TYPE_RANK)

    def summary(self):
        """Return a JSON-serializable summary of the notice, without the payload.

        Returns
        -------
        dict
            Fields describing the notice, for the store index and history log.
        """
        return {
            "topic": self.topic,
            "category": self.category,
            "source": self.source,
            "event_id": self.event_id,
            "event_names": list(self.event_names),
            "related_ids": list(self.related_ids),
            "alert_type": self.alert_type,
            "record_number": self.record_number,
            "trigger_time": self.trigger_time,
            "notice_time": self.notice_time,
            "is_retraction": self.is_retraction,
        }


def normalize_alert_type(value):
    """Lower-case an alert type so IGWN's "PRELIMINARY" and GCN's "initial" compare alike.

    Parameters
    ----------
    value : str or None
        Raw ``alert_type`` field.

    Returns
    -------
    str or None
        Lower-cased alert type, or None if absent.
    """
    if value is None or value == "":
        return None
    return str(value).strip().lower()


def localization_fields(obj):
    """Return the mapping that holds a localization's fields.

    IceCube's LVK track-search schema ``$ref``s core/Localization into each coincident
    event, which puts ``ra``/``dec``/``ra_dec_error`` directly on the event object, and the
    IceCube mission docs describe them that way too. The published example instead nests
    them under a ``localization`` key, and because the schema sets no
    ``unevaluatedProperties: false`` that example still validates. Accept either shape.

    Parameters
    ----------
    obj : dict
        Event or notice object that may hold localization fields flat or nested.

    Returns
    -------
    dict
        The mapping to read localization fields from.
    """
    nested = obj.get("localization")
    if isinstance(nested, dict):
        return {**obj, **nested}
    return obj


def parse_localization(obj, label=None):
    """Build a Localization from a gcn-schema core/Localization-shaped mapping.

    Parameters
    ----------
    obj : dict
        Mapping holding some of ``ra``, ``dec``, ``ra_dec_error``,
        ``containment_probability``, ``systematic_included``, ``healpix_file``,
        ``healpix_url``.
    label : str, optional
        Sub-event key to attach, for notices carrying several localizations.

    Returns
    -------
    Localization or None
        None if the mapping holds neither a position nor a map, so callers can tell
        "no localization" from "localization at the origin".
    """
    if not isinstance(obj, dict):
        return None
    fields = localization_fields(obj)
    ra = as_float(fields.get("ra"))
    dec = as_float(fields.get("dec"))
    semi_major, semi_minor, position_angle = parse_ra_dec_error(fields.get("ra_dec_error"))
    containment = as_float(fields.get("containment_probability"))
    healpix_url = fields.get("healpix_url") or None
    healpix_bytes = decode_healpix_file(fields.get("healpix_file"))
    if ra is None and dec is None and healpix_url is None and healpix_bytes is None:
        return None
    return Localization(
        label=label,
        ra=ra,
        dec=dec,
        semi_major_deg=semi_major,
        semi_minor_deg=semi_minor,
        position_angle_deg=position_angle,
        containment_probability=(DEFAULT_CONTAINMENT_PROBABILITY if containment is None else containment),
        systematic_included=bool(fields.get("systematic_included", DEFAULT_SYSTEMATIC_INCLUDED)),
        healpix_bytes=healpix_bytes,
        healpix_url=str(healpix_url) if healpix_url else None,
    )


def parse_record_number(payload):
    """Read core/Reporter's ``record_number`` sequence field as an int.

    Parameters
    ----------
    payload : dict
        Parsed notice.

    Returns
    -------
    int or None
        The sequence number, or None if absent or non-numeric.
    """
    value = as_float(payload.get("record_number"))
    return None if value is None else int(value)


def parse_igwn_gwalert(payload):
    """Normalize an ``igwn.gwalert`` notice.

    The localization is only ever a base64 multi-order FITS map under ``event.skymap``; the
    schema has no RA/Dec fields. A retraction arrives with ``alert_type`` "RETRACTION" and
    both ``event`` and ``external_coinc`` set to null, so there is nothing to localize. There
    is no sequence number on this topic, so ordering falls back to ``time_created``.

    Parameters
    ----------
    payload : dict
        Parsed notice.

    Returns
    -------
    dict
        Keyword arguments for NoticeRecord.
    """
    alert_type = normalize_alert_type(payload.get("alert_type"))
    event = payload.get("event") or {}
    external = payload.get("external_coinc") or {}
    superevent_id = payload.get("superevent_id") or UNKNOWN_EVENT_ID

    localizations = []
    skymap_bytes = decode_healpix_file(event.get("skymap"))
    if skymap_bytes is not None:
        localizations.append(Localization(healpix_bytes=skymap_bytes))
    combined_bytes = decode_healpix_file(external.get("combined_skymap"))
    if combined_bytes is not None:
        localizations.append(Localization(label=COMBINED_SKYMAP_LABEL, healpix_bytes=combined_bytes))

    related = as_id_tuple(external.get("gcn_notice_id")) + as_id_tuple(external.get("ivorn"))
    return {
        "event_id": str(superevent_id),
        "event_names": (str(superevent_id),),
        "related_ids": related,
        "alert_type": alert_type,
        "record_number": None,
        "trigger_time": event.get("time"),
        "notice_time": payload.get("time_created"),
        "is_retraction": alert_type == ALERT_TYPE_RETRACTION,
        "localizations": tuple(localizations),
    }


def parse_core_notice(payload):
    """Normalize a single-localization ``gcn.notices.*`` notice.

    Covers Einstein Probe WXT, Swift BAT-GUANO and the IceCube gold/bronze track alerts,
    which all compose core/Localization directly onto the notice. Swift GUANO's first
    record carries no localization at all and its later records add one, which is why a
    localization-free notice is normal here rather than an error.

    Parameters
    ----------
    payload : dict
        Parsed notice.

    Returns
    -------
    dict
        Keyword arguments for NoticeRecord.
    """
    alert_type = normalize_alert_type(payload.get("alert_type"))
    names = as_id_tuple(payload.get("event_name")) + as_id_tuple(payload.get("id"))
    localization = parse_localization(payload)
    follow_up = as_id_tuple(payload.get("follow_up_event")) + as_id_tuple(payload.get("ref_ID"))
    return {
        "event_id": names[0] if names else UNKNOWN_EVENT_ID,
        "event_names": names,
        "related_ids": follow_up,
        "alert_type": alert_type,
        "record_number": parse_record_number(payload),
        "trigger_time": payload.get("trigger_time"),
        "notice_time": payload.get("alert_datetime"),
        "is_retraction": alert_type == ALERT_TYPE_RETRACTION,
        "localizations": () if localization is None else (localization,),
    }


def parse_icecube_lvk_nu_track_search(payload):
    """Normalize a ``gcn.notices.icecube.lvk_nu_track_search`` notice.

    This topic has no identifier of its own: it reports the neutrinos found in the +/-500 s
    window around an LVK superevent, so it is keyed by ``ref_ID``, the superevent it
    followed up. Each entry in ``coincident_events`` carries its own error circle, labelled
    by the neutrino's ``RUNID_EVENTID``.

    Parameters
    ----------
    payload : dict
        Parsed notice.

    Returns
    -------
    dict
        Keyword arguments for NoticeRecord.
    """
    alert_type = normalize_alert_type(payload.get("alert_type"))
    reference_ids = as_id_tuple(payload.get("ref_ID"))
    localizations = []
    for index, event in enumerate(payload.get("coincident_events") or []):
        if not isinstance(event, dict):
            continue
        event_ids = as_id_tuple(event.get("id"))
        label = event_ids[0] if event_ids else f"event{index:02d}"
        localization = parse_localization(event, label=label)
        if localization is not None:
            localizations.append(localization)
    return {
        "event_id": reference_ids[0] if reference_ids else UNKNOWN_EVENT_ID,
        "event_names": reference_ids,
        "related_ids": reference_ids,
        "alert_type": alert_type,
        "record_number": parse_record_number(payload),
        "trigger_time": payload.get("trigger_time"),
        "notice_time": payload.get("alert_datetime"),
        "is_retraction": alert_type == ALERT_TYPE_RETRACTION,
        "localizations": tuple(localizations),
    }


def parse_boom_notice(payload):
    """Normalize a ``gcn.notices.boom.alert`` notice.

    BOOM is a ZTF/Rubin broker rather than a mission, so a notice is a container of optical
    targets and photometry with no top-level event id or trigger time. Each target carries
    an arcsecond-scale position and a ``gcn_crossmatch`` list naming the GW or GRB event it
    was matched to, which is the join key back to the rest of the store. The notice is filed
    under its first target's name, with the remaining targets recorded as event_names.

    Parameters
    ----------
    payload : dict
        Parsed notice.

    Returns
    -------
    dict
        Keyword arguments for NoticeRecord.
    """
    alert_type = normalize_alert_type(payload.get("alert_type"))
    data = payload.get("data") or {}
    names = []
    related = []
    localizations = []
    for index, target in enumerate(data.get("targets") or []):
        if not isinstance(target, dict):
            continue
        target_names = as_id_tuple(target.get("event_name")) + as_id_tuple(target.get("id"))
        label = target_names[0] if target_names else f"target{index:02d}"
        names.extend(target_names)
        for match in target.get("gcn_crossmatch") or []:
            if isinstance(match, dict):
                related.extend(as_id_tuple(match.get("ref_ID")))
        localization = parse_localization(target, label=label)
        if localization is not None:
            localizations.append(localization)

    photometry = data.get("photometry") or []
    first_observation = None
    for point in photometry:
        if isinstance(point, dict) and point.get("observation_start"):
            first_observation = point["observation_start"]
            break

    return {
        "event_id": names[0] if names else UNKNOWN_EVENT_ID,
        "event_names": tuple(dict.fromkeys(names)),
        "related_ids": tuple(dict.fromkeys(related)),
        "alert_type": alert_type,
        "record_number": parse_record_number(payload),
        "trigger_time": first_observation,
        "notice_time": payload.get("alert_datetime"),
        "is_retraction": alert_type == ALERT_TYPE_RETRACTION,
        "localizations": tuple(localizations),
    }


# Topic -> parser. Topics absent from this mapping fall back to parse_core_notice, which is
# the shape every gcn.notices.* topic composes, so a newly added mission topic still stores
# something useful instead of raising.
NOTICE_PARSERS = {
    TOPIC_IGWN_GWALERT: parse_igwn_gwalert,
    TOPIC_EINSTEIN_PROBE_WXT: parse_core_notice,
    TOPIC_SWIFT_BAT_GUANO: parse_core_notice,
    TOPIC_ICECUBE_GOLD_BRONZE: parse_core_notice,
    TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH: parse_icecube_lvk_nu_track_search,
    TOPIC_BOOM: parse_boom_notice,
}


def parse_notice(topic, raw):
    """Parse a raw Kafka message body into a NoticeRecord.

    Parameters
    ----------
    topic : str
        Kafka topic the message arrived on, which selects the parser and the store route.
    raw : bytes or str or dict
        Message body: JSON bytes as delivered by Kafka, decoded JSON text, or an
        already-parsed payload.

    Returns
    -------
    NoticeRecord
        The normalized notice.

    Raises
    ------
    ValueError
        If the body is not valid JSON, or is valid JSON but not an object.
    """
    if isinstance(raw, dict):
        payload = raw
    else:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"notice on {topic} is not a JSON object: {type(payload).__name__}")

    category, source = TOPIC_ROUTING.get(topic, (CATEGORY_GRB, "unknown"))
    parser = NOTICE_PARSERS.get(topic, parse_core_notice)
    parsed = parser(payload)
    return NoticeRecord(topic=topic, category=category, source=source, payload=payload, **parsed)


def notice_instruments(record):
    """Describe the reporting instrument for a notice, for the synthesized map's header.

    Parameters
    ----------
    record : NoticeRecord
        Normalized notice.

    Returns
    -------
    list of str
        One-element list naming mission and instrument where the notice reports them, or an
        empty list if it reports neither.
    """
    payload = record.payload
    parts = [str(payload[key]) for key in ("mission", "instrument") if payload.get(key)]
    if not parts:
        return []
    return ["/".join(dict.fromkeys(parts))]
