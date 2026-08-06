"""Tests for the GCN Kafka listener loop.

No test here touches a network or imports a Kafka client: FakeConsumer stands in for
gcn_kafka.Consumer, which is enough because the properties worth pinning down are all about
what the loop does with the messages it is handed.
"""

import json
import threading

import pytest
from desi_aap import gcn_listener, gcn_notices, gcn_store
from gcn_examples import GUANO_TRIGGER_ID, igwn_gwalert, swift_bat_guano
from test_gcn_store import fake_resolve


class FakeError:
    """Stand-in for the KafkaError returned by Message.error()."""

    def __init__(self, description):
        self.description = description

    def __str__(self):
        """Return the error description."""
        return self.description


class FakeMessage:
    """Stand-in for a confluent_kafka Message.

    Parameters
    ----------
    topic : str
        Topic the message arrived on.
    value : bytes or None
        Message body.
    error : FakeError, optional
        Broker-side error, for messages that carry no payload.
    """

    def __init__(self, topic, value, error=None):
        self._topic = topic
        self._value = value
        self._error = error

    def topic(self):
        """Return the topic."""
        return self._topic

    def value(self):
        """Return the message body."""
        return self._value

    def error(self):
        """Return the broker-side error, or None for a proper message."""
        return self._error


class FakeConsumer:
    """Stand-in for gcn_kafka.Consumer, driven by a list of batches.

    Parameters
    ----------
    batches : list of list of FakeMessage
        Batches to hand out, one per consume() call. Once exhausted, consume() returns empty
        lists forever, which is what a live consumer does when the buffer is drained.
    """

    def __init__(self, batches):
        self._batches = list(batches)
        self.committed = []
        self.closed = False
        self.consume_calls = 0

    def consume(self, num_messages, timeout):
        """Return the next batch, or an empty list once they are exhausted."""
        self.consume_calls += 1
        return self._batches.pop(0) if self._batches else []

    def commit(self, message):
        """Record a committed offset."""
        self.committed.append(message)

    def close(self):
        """Record that the consumer was closed."""
        self.closed = True


def notice_message(payload, topic):
    """Build a FakeMessage carrying a JSON notice.

    Parameters
    ----------
    payload : dict
        Notice payload.
    topic : str
        Topic to attach.

    Returns
    -------
    FakeMessage
        The message.
    """
    return FakeMessage(topic, json.dumps(payload).encode("utf-8"))


def test_credentials_must_be_present_and_the_error_names_both_variables():
    """A missing secret is the most likely first-run failure, so the message has to be useful."""
    with pytest.raises(RuntimeError) as failure:
        gcn_listener.credentials_from_env(environ={})
    message = str(failure.value)
    assert gcn_listener.CLIENT_ID_ENV_VAR in message
    assert gcn_listener.CLIENT_SECRET_ENV_VAR in message
    assert "gcn.nasa.gov/quickstart" in message


def test_credentials_are_read_from_the_environment():
    """Credentials never belong in the source, so they come from the environment only."""
    environ = {
        gcn_listener.CLIENT_ID_ENV_VAR: "an-id",
        gcn_listener.CLIENT_SECRET_ENV_VAR: "a-secret",
    }
    assert gcn_listener.credentials_from_env(environ=environ) == ("an-id", "a-secret")


def test_an_empty_credential_counts_as_missing():
    """An exported-but-empty variable is the confusing case; treat it as absent."""
    environ = {gcn_listener.CLIENT_ID_ENV_VAR: "", gcn_listener.CLIENT_SECRET_ENV_VAR: "x"}
    with pytest.raises(RuntimeError, match=gcn_listener.CLIENT_ID_ENV_VAR):
        gcn_listener.credentials_from_env(environ=environ)


def test_consumer_config_sets_a_durable_group_and_manual_commits():
    """gcn-kafka invents a random group id when none is set, which silently loses notices."""
    config = gcn_listener.consumer_config()
    assert config["group.id"] == gcn_listener.CONSUMER_GROUP_ID
    assert config["group.id"]
    assert config["enable.auto.commit"] is False
    assert config["auto.offset.reset"] == "earliest"
    assert gcn_listener.consumer_config(from_earliest=False)["auto.offset.reset"] == "latest"


def test_handle_message_stores_the_notice(tmp_path):
    """The end-to-end path from a raw message body to files on disk."""
    payload = swift_bat_guano(3)
    entry = gcn_listener.handle_message(
        gcn_notices.TOPIC_SWIFT_BAT_GUANO,
        json.dumps(payload).encode("utf-8"),
        root=tmp_path,
        resolve=fake_resolve(),
    )
    assert entry["stored"] is True
    assert (tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID).is_dir()


def test_batch_commits_each_message_only_after_storing_it(tmp_path):
    """Committing before the write would drop a notice on any crash in between."""
    consumer = FakeConsumer(
        [
            [
                notice_message(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO),
                notice_message(igwn_gwalert(), gcn_notices.TOPIC_IGWN_GWALERT),
            ]
        ]
    )
    counts = gcn_listener.process_batch(consumer, root=tmp_path, resolve=fake_resolve())
    assert counts == {"consumed": 2, "stored": 2, "skipped": 0, "failed": 0, "errors": 0}
    assert len(consumer.committed) == 2
    assert len(gcn_store.iter_index(root=tmp_path)) == 2


def test_replayed_message_is_committed_but_not_stored_twice(tmp_path):
    """At-least-once delivery means duplicates are normal, not an error to report."""
    message = notice_message(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO)
    consumer = FakeConsumer([[message, message]])
    counts = gcn_listener.process_batch(consumer, root=tmp_path, resolve=fake_resolve())
    assert counts["stored"] == 1
    assert counts["skipped"] == 1
    # Both offsets are still committed, or the duplicate would be redelivered forever.
    assert len(consumer.committed) == 2
    assert len(gcn_store.iter_index(root=tmp_path)) == 1


def test_broker_errors_are_logged_and_skipped_without_committing(tmp_path):
    """Partition EOF arrives as an error message with no payload; it is not a notice."""
    consumer = FakeConsumer(
        [[FakeMessage(gcn_notices.TOPIC_BOOM, None, error=FakeError("Broker: partition EOF"))]]
    )
    counts = gcn_listener.process_batch(consumer, root=tmp_path, resolve=fake_resolve())
    assert counts == {"consumed": 1, "stored": 0, "skipped": 0, "failed": 0, "errors": 1}
    assert consumer.committed == []
    assert not list(tmp_path.iterdir())


def test_unparseable_message_is_quarantined_and_committed(tmp_path):
    """One bad payload must not stall every notice queued behind it."""
    consumer = FakeConsumer([[FakeMessage(gcn_notices.TOPIC_BOOM, b"{not json")]])
    counts = gcn_listener.process_batch(consumer, root=tmp_path, resolve=fake_resolve())
    assert counts["failed"] == 1
    # Committed anyway: the alternative is replaying the same failure forever.
    assert len(consumer.committed) == 1
    quarantined = list((tmp_path / gcn_store.QUARANTINE_SUBDIR).glob("*.payload"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"{not json"


def test_a_failure_writing_a_map_is_quarantined_rather_than_lost(tmp_path):
    """A synthesis or filesystem failure is still a payload we must not silently drop."""

    def explode(path_stem, localization, record):
        raise OSError("disk on fire")

    consumer = FakeConsumer([[notice_message(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO)]])
    counts = gcn_listener.process_batch(consumer, root=tmp_path, resolve=explode)
    assert counts["failed"] == 1
    errors = list((tmp_path / gcn_store.QUARANTINE_SUBDIR).glob("*.error.json"))
    assert "disk on fire" in json.loads(errors[0].read_text())["error"]


def test_a_batch_continues_past_a_bad_message(tmp_path):
    """The message after a poison pill has to be processed in the same batch."""
    consumer = FakeConsumer(
        [
            [
                FakeMessage(gcn_notices.TOPIC_BOOM, b"garbage"),
                notice_message(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO),
            ]
        ]
    )
    counts = gcn_listener.process_batch(consumer, root=tmp_path, resolve=fake_resolve())
    assert counts["failed"] == 1
    assert counts["stored"] == 1
    assert len(consumer.committed) == 2


def test_run_listener_once_drains_the_buffer_and_stops(tmp_path):
    """The catch-up mode: process what is buffered, then exit rather than block."""
    consumer = FakeConsumer(
        [
            [notice_message(swift_bat_guano(1), gcn_notices.TOPIC_SWIFT_BAT_GUANO)],
            [notice_message(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO)],
        ]
    )
    totals = gcn_listener.run_listener(root=tmp_path, once=True, consumer=consumer, resolve=fake_resolve())
    assert totals["consumed"] == 2
    assert totals["stored"] == 2
    # Three calls: two batches, then the empty one that ends the drain.
    assert consumer.consume_calls == 3
    # The caller supplied the consumer, so the caller still owns closing it.
    assert consumer.closed is False


def test_run_listener_stops_when_the_stop_event_is_set(tmp_path):
    """SIGTERM sets this event, and the loop must notice it rather than run forever."""
    stop = threading.Event()
    stop.set()
    totals = gcn_listener.run_listener(
        root=tmp_path, consumer=FakeConsumer([]), stop=stop, resolve=fake_resolve()
    )
    assert totals["consumed"] == 0


def test_run_listener_keeps_going_until_asked_to_stop(tmp_path):
    """Without --once the loop is indefinite, which is what a daemon needs."""
    stop = threading.Event()

    class StoppingConsumer(FakeConsumer):
        def consume(self, num_messages, timeout):
            batch = super().consume(num_messages, timeout)
            if self.consume_calls >= 3:
                stop.set()
            return batch

    consumer = StoppingConsumer([[notice_message(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO)]])
    totals = gcn_listener.run_listener(root=tmp_path, consumer=consumer, stop=stop, resolve=fake_resolve())
    # Two empty batches after the first did not end the loop; only the event did.
    assert consumer.consume_calls == 3
    assert totals["stored"] == 1


def test_install_shutdown_handlers_returns_an_event_set_by_the_signal():
    """A daemon has to shut down on SIGTERM without dropping the message in hand."""
    import signal

    original = {number: signal.getsignal(number) for number in gcn_listener.SHUTDOWN_SIGNALS}
    try:
        stop = gcn_listener.install_shutdown_handlers()
        assert not stop.is_set()
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        assert stop.is_set()
    finally:
        for number, previous in original.items():
            signal.signal(number, previous)


def test_make_consumer_requires_credentials_before_importing_a_kafka_client(monkeypatch):
    """Failing on configuration before touching the network keeps the error legible."""
    monkeypatch.delenv(gcn_listener.CLIENT_ID_ENV_VAR, raising=False)
    monkeypatch.delenv(gcn_listener.CLIENT_SECRET_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=gcn_listener.CLIENT_ID_ENV_VAR):
        gcn_listener.make_consumer()


def test_default_topics_cover_gw_grb_and_neutrino_sources():
    """The listener subscribes to everything routed, so a new topic is live once routed."""
    assert gcn_listener.DEFAULT_TOPICS == gcn_notices.DEFAULT_TOPICS
    assert gcn_notices.TOPIC_IGWN_GWALERT in gcn_listener.DEFAULT_TOPICS
    assert len(gcn_listener.DEFAULT_TOPICS) == 6
