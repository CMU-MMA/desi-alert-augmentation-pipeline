"""Long-running GCN Kafka listener that files localizations into the store.

GraceDB gives us GW skymaps after the fact; this gives us them, and the GRB and neutrino
localizations, as they are published. The loop is deliberately dull: consume, parse, store,
commit, and never let one bad message stop the stream.

Two properties are worth stating outright, because both are easy to get wrong and expensive
to discover later.

Durable offsets. gcn-kafka assigns a random UUID group id when none is configured, so a
consumer built with defaults starts from scratch on every restart and silently loses whatever
arrived while it was down. This module always sets an explicit group id, disables auto-commit,
and commits each message only after it has been stored, which makes delivery at-least-once
rather than at-most-once. GCN's buffers only hold the past few days, so a listener that stays
down longer than that has a gap no restart can fill; use the GraceDB path in
desi_aap.gracedb_tools to backfill GW events, and GCN's notices archive for the rest.

Poison messages. A payload that cannot be parsed or stored is quarantined under the store
root and its offset is committed anyway. Not committing would replay the same bad message
forever and block every later notice behind it.
"""

import logging
import os
import signal
import threading

from desi_aap.gcn_notices import DEFAULT_TOPICS, parse_notice
from desi_aap.gcn_store import STORE_ROOT, quarantine_payload, store_notice

LOGGER = logging.getLogger(__name__)

# GCN deployment to connect to. The test and dev domains carry the same schemas with
# synthetic events, which is the safe place to point a new consumer group.
GCN_DOMAIN_PRODUCTION = "gcn.nasa.gov"
GCN_DOMAIN_TEST = "test.gcn.nasa.gov"
GCN_DOMAIN = GCN_DOMAIN_PRODUCTION

# Kafka consumer group. This is the identity GCN's brokers store our offsets against, so it
# must stay stable across restarts and must differ from any other consumer we run: two
# processes sharing a group id split the partitions between them rather than each seeing
# every notice.
CONSUMER_GROUP_ID = "desi-aap-localizations"

# Where a new group starts. "earliest" picks up everything still in GCN's buffer, so a
# first run backfills the past few days instead of only seeing what arrives next.
AUTO_OFFSET_RESET = "earliest"

# Offsets are committed by hand after each message is stored; see the module docstring.
ENABLE_AUTO_COMMIT = False

# consume() returns up to this many messages per call. The timeout also bounds how long the
# loop blocks, which is what lets a Ctrl-C or a SIGTERM be noticed: consume() with no timeout
# ignores Ctrl-C entirely.
CONSUME_BATCH_SIZE = 20
CONSUME_TIMEOUT_S = 1.0

# Credentials come from the environment, and these names match the repository secrets of the
# same names, following the same pattern as the TNS credentials. GCN publishes no
# environment-variable convention of its own; get the values from
# https://gcn.nasa.gov/quickstart. Nothing in the test suite needs them, because every test
# drives the loop through a fake consumer rather than a connection.
CLIENT_ID_ENV_VAR = "GCN_CLIENT_ID"
CLIENT_SECRET_ENV_VAR = "GCN_CLIENT_SECRET"

# Signals that ask the listener to stop after finishing the message in hand.
SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def credentials_from_env(environ=None):
    """Read the GCN client credentials from the environment.

    Parameters
    ----------
    environ : mapping, optional
        Environment to read; defaults to os.environ.

    Returns
    -------
    tuple of str
        ``(client_id, client_secret)``.

    Raises
    ------
    RuntimeError
        If either variable is unset or empty, naming both so the fix is obvious.
    """
    environ = os.environ if environ is None else environ
    client_id = environ.get(CLIENT_ID_ENV_VAR)
    client_secret = environ.get(CLIENT_SECRET_ENV_VAR)
    missing = [
        name
        for name, value in ((CLIENT_ID_ENV_VAR, client_id), (CLIENT_SECRET_ENV_VAR, client_secret))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"missing GCN credentials in environment: {', '.join(missing)}. "
            f"Create a client at https://gcn.nasa.gov/quickstart and export both."
        )
    return client_id, client_secret


def consumer_config(group_id=CONSUMER_GROUP_ID, from_earliest=True):
    """Build the confluent-kafka configuration for a durable listener.

    Parameters
    ----------
    group_id : str, optional
        Kafka consumer group id; see CONSUMER_GROUP_ID.
    from_earliest : bool, optional
        Whether a group with no committed offset starts at the oldest buffered message rather
        than the newest.

    Returns
    -------
    dict
        Configuration to pass to gcn_kafka.Consumer.
    """
    return {
        "group.id": group_id,
        "auto.offset.reset": AUTO_OFFSET_RESET if from_earliest else "latest",
        "enable.auto.commit": ENABLE_AUTO_COMMIT,
    }


def make_consumer(
    topics=DEFAULT_TOPICS,
    group_id=CONSUMER_GROUP_ID,
    domain=GCN_DOMAIN,
    from_earliest=True,
    environ=None,
    config=None,
):
    """Connect a GCN Kafka consumer and subscribe it to the given topics.

    gcn_kafka is imported here rather than at module scope so that parsing, synthesizing and
    storing notices all work in an environment without a Kafka client installed.

    Parameters
    ----------
    topics : sequence of str, optional
        Topics to subscribe to; defaults to every topic in TOPIC_ROUTING.
    group_id : str, optional
        Kafka consumer group id.
    domain : str, optional
        GCN domain; see GCN_DOMAIN.
    from_earliest : bool, optional
        Passed to consumer_config().
    environ : mapping, optional
        Environment to read credentials from.
    config : dict, optional
        Extra confluent-kafka settings, merged over the defaults.

    Returns
    -------
    gcn_kafka.Consumer
        A subscribed consumer.
    """
    from gcn_kafka import Consumer

    client_id, client_secret = credentials_from_env(environ=environ)
    settings = consumer_config(group_id=group_id, from_earliest=from_earliest)
    if config:
        settings.update(config)
    consumer = Consumer(settings, client_id=client_id, client_secret=client_secret, domain=domain)
    consumer.subscribe(list(topics))
    LOGGER.info("subscribed to %d topics as group %r on %s", len(topics), group_id, domain)
    return consumer


def handle_message(topic, raw, root=STORE_ROOT, resolve=None):
    """Parse and store one notice body.

    Parameters
    ----------
    topic : str
        Topic the message arrived on.
    raw : bytes or str
        Message body exactly as received.
    root : pathlib.Path, optional
        Store root.
    resolve : callable, optional
        Skymap writer, passed through to store_notice().

    Returns
    -------
    dict
        The store's history entry for the notice.
    """
    record = parse_notice(topic, raw)
    entry = store_notice(record, raw, root=root, resolve=resolve)
    if entry.get("stored"):
        LOGGER.info(
            "stored %s %s (%s, %d map(s)%s)",
            record.category,
            record.event_id,
            record.alert_type or "no alert_type",
            len(entry.get("skymaps") or []),
            ", RETRACTED" if record.is_retraction else "",
        )
    else:
        LOGGER.debug("skipped redelivery of %s %s", record.category, record.event_id)
    return entry


def install_shutdown_handlers(signals=SHUTDOWN_SIGNALS):
    """Install signal handlers that ask the listener to stop cleanly.

    Parameters
    ----------
    signals : sequence, optional
        Signals to handle; see SHUTDOWN_SIGNALS.

    Returns
    -------
    threading.Event
        Set when one of the signals arrives.
    """
    stop = threading.Event()

    def request_stop(signal_number, _frame):
        LOGGER.info("received signal %s, finishing current message then stopping", signal_number)
        stop.set()

    for number in signals:
        signal.signal(number, request_stop)
    return stop


def process_batch(consumer, root=STORE_ROOT, resolve=None, batch=None):
    """Consume one batch of messages, storing each and committing its offset.

    An offset is committed after the message is stored, so a crash mid-batch replays the last
    message rather than dropping it; store_notice() recognizes the redelivery by its payload
    digest and does nothing. A message that fails to parse or store is quarantined and
    committed anyway, so it cannot block the messages behind it.

    Parameters
    ----------
    consumer : gcn_kafka.Consumer
        Subscribed consumer.
    root : pathlib.Path, optional
        Store root.
    resolve : callable, optional
        Skymap writer, passed through to handle_message().
    batch : list, optional
        Messages to process instead of calling consumer.consume(), for testing.

    Returns
    -------
    dict
        Counts of "consumed", "stored", "skipped", "failed" and "errors" (broker-side message
        errors, which are logged and not quarantined because they carry no payload).
    """
    messages = consumer.consume(CONSUME_BATCH_SIZE, CONSUME_TIMEOUT_S) if batch is None else batch
    counts = {"consumed": 0, "stored": 0, "skipped": 0, "failed": 0, "errors": 0}
    for message in messages:
        counts["consumed"] += 1
        error = message.error()
        if error is not None:
            # Broker-side conditions arrive as messages rather than exceptions, and include
            # benign ones such as partition EOF, so log and carry on rather than stopping.
            LOGGER.warning("kafka message error: %s", error)
            counts["errors"] += 1
            continue
        topic = message.topic()
        raw = message.value()
        try:
            entry = handle_message(topic, raw, root=root, resolve=resolve)
        # Deliberately broad: a malformed notice, a failed synthesis and a full disk all have
        # to be survivable, because the messages behind this one are still worth storing.
        except Exception as failure:
            path = quarantine_payload(topic, raw, failure, root=root)
            LOGGER.exception("quarantined unparseable notice from %s to %s", topic, path)
            counts["failed"] += 1
        else:
            counts["stored" if entry.get("stored") else "skipped"] += 1
        consumer.commit(message)
    return counts


def run_listener(
    root=STORE_ROOT,
    topics=DEFAULT_TOPICS,
    group_id=CONSUMER_GROUP_ID,
    domain=GCN_DOMAIN,
    from_earliest=True,
    once=False,
    consumer=None,
    stop=None,
    resolve=None,
):
    """Run the listener until it is asked to stop.

    Parameters
    ----------
    root : pathlib.Path, optional
        Store root.
    topics : sequence of str, optional
        Topics to subscribe to.
    group_id : str, optional
        Kafka consumer group id.
    domain : str, optional
        GCN domain.
    from_earliest : bool, optional
        Whether a new group starts at the oldest buffered message.
    once : bool, optional
        Drain the buffer until a batch comes back empty, then return, instead of running
        indefinitely. Useful for a scheduled catch-up run or a smoke test.
    consumer : object, optional
        An already-connected consumer, used instead of building one.
    stop : threading.Event, optional
        Event that ends the loop when set. Defaults to one wired to SIGINT and SIGTERM, unless
        a consumer was supplied, in which case the caller owns shutdown.
    resolve : callable, optional
        Skymap writer, passed through to process_batch().

    Returns
    -------
    dict
        Cumulative counts as described in process_batch().
    """
    owns_consumer = consumer is None
    if owns_consumer:
        consumer = make_consumer(topics=topics, group_id=group_id, domain=domain, from_earliest=from_earliest)
        if stop is None:
            stop = install_shutdown_handlers()
    if stop is None:
        stop = threading.Event()

    totals = {"consumed": 0, "stored": 0, "skipped": 0, "failed": 0, "errors": 0}
    try:
        while not stop.is_set():
            counts = process_batch(consumer, root=root, resolve=resolve)
            for key, value in counts.items():
                totals[key] += value
            if once and counts["consumed"] == 0:
                break
    finally:
        if owns_consumer:
            consumer.close()
    LOGGER.info(
        "listener finished: %d consumed, %d stored, %d redelivered, %d quarantined, %d errors",
        totals["consumed"],
        totals["stored"],
        totals["skipped"],
        totals["failed"],
        totals["errors"],
    )
    return totals
