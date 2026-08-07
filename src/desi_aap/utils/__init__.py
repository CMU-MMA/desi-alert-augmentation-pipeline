"""Small helpers shared across the package."""

from datetime import UTC, datetime

# Sortable UTC timestamp used to name everything one run produces: each stage's
# output file, and the run's log.
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def run_stamp() -> str:
    """Timestamp naming one run's output, taken when the run starts.

    Generated once per invocation and passed down, so the log and every stage's
    output file carry the same one and a run's artifacts group by name.

    Returns
    -------
    str
        The current UTC time, formatted as :data:`STAMP_FORMAT`.
    """
    return datetime.now(UTC).strftime(STAMP_FORMAT)
