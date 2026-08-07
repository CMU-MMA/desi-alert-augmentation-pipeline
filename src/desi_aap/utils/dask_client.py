"""Dask client management for pipeline stages."""

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def dask_client(client_kwargs: dict[str, Any] | None = None) -> Iterator[Any]:
    """Run a block against a Dask distributed client, or the default scheduler.

    An empty or absent ``client_kwargs`` yields ``None`` and starts no cluster,
    which leaves LSDB on dask's default threaded scheduler. That keeps small
    runs -- and the test suite -- free of the cost of spinning up workers, so
    the ``[dask]`` config section is opt-in rather than mandatory.

    Parameters
    ----------
    client_kwargs : dict, optional
        Keyword arguments for :class:`dask.distributed.Client`.
        ``local_directory`` defaults to a temporary directory that is removed
        on exit, so workers do not scatter spill files into the working
        directory.

    Yields
    ------
    dask.distributed.Client or None
        The live client, or None when no cluster was requested.
    """
    kwargs = dict(client_kwargs or {})
    if not kwargs:
        logger.debug("No Dask settings given; using dask's default scheduler.")
        yield None
        return

    from dask.distributed import Client

    scratch = None
    if "local_directory" not in kwargs:
        scratch = tempfile.TemporaryDirectory()
        kwargs["local_directory"] = scratch.name

    client = Client(**kwargs)
    try:
        logger.info("Dask dashboard: %s", client.dashboard_link)
        yield client
    finally:
        client.close()
        if scratch is not None:
            scratch.cleanup()
