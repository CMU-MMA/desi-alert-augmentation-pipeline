#!/usr/bin/env python
"""Listen to GCN Kafka and file localization maps into the store.

Set the credentials from https://gcn.nasa.gov/quickstart first:

    export GCN_CLIENT_ID=...
    export GCN_CLIENT_SECRET=...

Then run it as a daemon:

    python scripts/listen_gcn.py --store-root /path/to/gcn_localizations

or as a catch-up pass that drains whatever is buffered and exits:

    python scripts/listen_gcn.py --once

Point --domain at test.gcn.nasa.gov to rehearse against GCN's synthetic events. Keep the
--group-id stable across restarts: it is the identity GCN stores our stream position against,
so changing it either replays the buffer or skips what arrived while we were down.
"""

import argparse
import logging
import sys
from pathlib import Path

from desi_aap.gcn_listener import (
    CONSUMER_GROUP_ID,
    GCN_DOMAIN,
    GCN_DOMAIN_TEST,
    run_listener,
)
from desi_aap.gcn_notices import DEFAULT_TOPICS
from desi_aap.gcn_store import STORE_ROOT

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args(argv=None):
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=STORE_ROOT,
        help=f"directory to write notices and maps into (default: {STORE_ROOT})",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=list(DEFAULT_TOPICS),
        metavar="TOPIC",
        help="Kafka topics to subscribe to (default: every routed topic)",
    )
    parser.add_argument(
        "--group-id",
        default=CONSUMER_GROUP_ID,
        help=f"Kafka consumer group id, keep stable across restarts (default: {CONSUMER_GROUP_ID})",
    )
    parser.add_argument(
        "--domain",
        default=GCN_DOMAIN,
        help=f"GCN domain, e.g. {GCN_DOMAIN_TEST} to rehearse (default: {GCN_DOMAIN})",
    )
    parser.add_argument(
        "--from-latest",
        action="store_true",
        help="for a group with no stored position, start at the newest message instead of "
        "backfilling GCN's buffer of the past few days",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="drain the buffered messages and exit, instead of listening indefinitely",
    )
    parser.add_argument(
        "--log-level", default=DEFAULT_LOG_LEVEL, choices=LOG_LEVELS, help="logging verbosity"
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the listener from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to sys.argv[1:].

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    try:
        run_listener(
            root=args.store_root,
            topics=args.topics,
            group_id=args.group_id,
            domain=args.domain,
            from_earliest=not args.from_latest,
            once=args.once,
        )
    except RuntimeError as error:
        # Missing credentials are the common case here, and a traceback would bury the fix.
        logging.getLogger(__name__).error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
