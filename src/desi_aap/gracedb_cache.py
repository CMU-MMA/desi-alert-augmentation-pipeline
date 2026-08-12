"""On-disk cache for the per-superevent GraceDB metadata.

A run of :func:`desi_aap.gracedb_tools.fetch_gracedb_superevents` makes three kinds of network
call: one paginated ``superevents()`` listing, then per superevent a ``files()`` listing and a
``p_astro.json`` download, then a skymap download. The listing is a handful of requests; the
per-superevent calls are two per superevent and dominate the total, so those are what is cached
here.

The superevent payload carries no modification timestamp, so freshness is judged from the mutable
fields the listing already returns -- see :func:`superevent_fingerprint` -- with an age backstop
for events too young to have settled. Everything is written atomically, because the pipeline runs
unattended and a half-written cache entry that is then trusted is worse than no cache at all.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Suffix given to the temporary file an atomic write goes through. Shared because
# GraceDbCache.read_entry must know which names to disregard: a run killed between the write and
# the rename leaves one of these behind, and it is not a cache entry.
TEMP_SUFFIX = ".tmp"

# Subdirectories of a cache. Shared so that a caller reading the cache by hand, or clearing part
# of it, does not have to rediscover the layout from the code that writes it.
ENTRY_SUBDIR = "superevents"
SKYMAP_SUBDIR = "skymaps"


def safe_file_part(value):
    """Sanitize a value for use as part of a local file name.

    Parameters
    ----------
    value : object
        Value to sanitize; converted with str() first.

    Returns
    -------
    str
        The string with each run of characters outside [A-Za-z0-9_.-] replaced by a single
        underscore.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def atomic_write_bytes(path, payload):
    """Write bytes to a path so that a reader never sees a partial file.

    The payload goes to a temporary file in the same directory, which is flushed and fsynced, and
    only then renamed over the destination. os.replace is atomic within a filesystem on POSIX, so
    the destination either does not exist or holds the whole payload; there is no window in which
    it holds a prefix of it.

    This matters because the caller's only validity check on a cached file is that it exists. A
    plain write leaves a truncated file behind when the process dies partway through -- a wall-clock
    limit or an evicted node -- and every later run then trusts it. The same-directory temporary
    file also makes two concurrent runs safe without a lock: a run's rename simply wins or is
    overwritten, and neither sees a torn file.

    Parameters
    ----------
    path : pathlib.Path
        Destination file. Its parent directory is created if it does not exist.
    payload : bytes
        Bytes to write.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=TEMP_SUFFIX)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path, obj):
    """Write an object as JSON through :func:`atomic_write_bytes`.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    obj : object
        Anything json.dumps accepts.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    return atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))


def superevent_fingerprint(superevent):
    """Summarize the fields of a superevent that signal its files may have changed.

    GraceDB's superevent payload carries no modification timestamp -- checked against the live
    production API, which returns created, t_start, t_0, t_end, far, labels and
    preferred_event_data, and nothing that tracks the record's own last update. So freshness is
    inferred from the mutable fields the listing already returns, which costs no extra request.

    Kept as a plain dict rather than a hash so that a cache entry explains itself when someone
    opens it, and so a stale entry shows which field moved.

    Parameters
    ----------
    superevent : dict
        Superevent dict from GraceDb.superevents().

    Returns
    -------
    dict
        The labels, sorted; the preferred event's graceid; the FAR; and t_0. Labels are sorted
        because GraceDB does not return them in a stable order, and an order-only difference is
        not a change worth re-downloading for.
    """
    preferred = superevent.get("preferred_event_data") or {}
    return {
        "labels": sorted(superevent.get("labels") or []),
        "preferred_event": preferred.get("graceid"),
        "far": superevent.get("far"),
        "t_0": superevent.get("t_0"),
    }


def latest_revision(files, basename):
    """Return the highest revision number a file listing holds for one file name.

    GraceDB exposes every revision of a file under a ",N" suffix and points the unsuffixed name at
    the newest one, so the unsuffixed name silently changes content over time. Observed on the
    live API: S250206dm lists Bilby.multiorder.fits alongside both Bilby.multiorder.fits,0 and
    Bilby.multiorder.fits,1. Recording this number is what lets a cached download notice it has
    been superseded.

    Parameters
    ----------
    files : iterable of str
        File names available for the superevent.
    basename : str
        Unversioned file name to look for, e.g. "bayestar.multiorder.fits". A name that is itself
        versioned has no revisions of its own and yields None, which is correct: it names one fixed
        revision that cannot change.

    Returns
    -------
    int or None
        The largest N across the "<basename>,N" entries, or None when the listing holds none.
    """
    prefix = f"{basename},"
    revisions = [
        int(name[len(prefix) :])
        for name in files
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
    ]
    return max(revisions) if revisions else None


@dataclass(frozen=True)
class GraceDbCache:
    """Where cached GraceDB metadata lives, and how stale an entry may get.

    Attributes
    ----------
    cache_dir : pathlib.Path
        Root of the cache. Required, with no default: a module-level default resolved against the
        working directory is what let the same skymaps be downloaded twice into two different
        notebook directories, so the location is always the caller's explicit decision. Set it from
        the ``[gracedb]`` section of the pipeline config; see
        :meth:`desi_aap.config.GraceDbConfig.to_cache`.
    recheck_window : datetime.timedelta, optional
        Superevents whose merger is more recent than this are re-checked on every run even when
        their fingerprint is unchanged, because a file can be uploaded without moving any field
        the listing reports. Older superevents are trusted until their fingerprint moves. Defaults
        to 30 days.
    """

    cache_dir: Path
    recheck_window: timedelta = timedelta(days=30)

    @property
    def entry_dir(self):
        """Directory holding one JSON entry per superevent."""
        return self.cache_dir / ENTRY_SUBDIR

    @property
    def skymap_dir(self):
        """Directory holding the downloaded skymaps."""
        return self.cache_dir / SKYMAP_SUBDIR

    def entry_path(self, superevent_id):
        """Path of one superevent's cache entry.

        Parameters
        ----------
        superevent_id : str
            Superevent identifier, e.g. "S190425z".

        Returns
        -------
        pathlib.Path
            The entry's location, whether or not it exists.
        """
        return self.entry_dir / f"{safe_file_part(superevent_id)}.json"

    def resolve(self, relative_path):
        """Turn a path stored in an entry into an absolute one.

        Entries record paths relative to ``cache_dir`` so that a cache stays valid when it is moved
        between a laptop, ``$HOME`` and a shared project filesystem.

        Parameters
        ----------
        relative_path : str or pathlib.Path or None
            Path as stored in the entry. None passes through.

        Returns
        -------
        pathlib.Path or None
            The absolute path, or None.
        """
        if not relative_path:
            return None
        return (self.cache_dir / relative_path).resolve()

    def read_entry(self, superevent_id):
        """Read one superevent's cache entry.

        An entry that is missing, empty, truncated, not valid JSON, or not a JSON object is
        reported as absent rather than raised. A cache is an optimization: any damage to it must
        cost a re-fetch, never a failed run.

        Parameters
        ----------
        superevent_id : str
            Superevent identifier.

        Returns
        -------
        dict or None
            The entry, or None if there is no usable one.
        """
        try:
            with self.entry_path(superevent_id).open("rb") as handle:
                entry = json.load(handle)
        except (OSError, ValueError):
            # ValueError covers json.JSONDecodeError, which is what an empty or truncated file
            # raises; OSError covers the missing file and an unreadable directory.
            return None
        return entry if isinstance(entry, dict) else None

    def write_entry(self, superevent_id, entry):
        """Write one superevent's cache entry atomically.

        Parameters
        ----------
        superevent_id : str
            Superevent identifier.
        entry : dict
            The entry to store.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        return atomic_write_json(self.entry_path(superevent_id), entry)

    def status(self, entry, superevent, *, gw_time, now=None):
        """Decide whether a cache entry can be used for a superevent as the listing now reports it.

        Parameters
        ----------
        entry : dict or None
            The cached entry, from :meth:`read_entry`. None or empty means nothing is cached.
        superevent : dict
            The superevent as the live listing reports it now.
        gw_time : pandas.Timestamp or pandas.NaT
            The superevent's merger time, used for the age backstop. NaT counts as recent, so a
            superevent whose time could not be read is re-checked rather than trusted forever.
        now : pandas.Timestamp, optional
            The current time, for the age comparison. Defaults to now, in UTC. Present so tests
            can place a superevent either side of the window without waiting.

        Returns
        -------
        str
            One of "hit" (use the entry), "miss" (nothing cached), "stale_fingerprint" (a listed
            field moved) or "stale_age" (too recent to trust). Every value other than "hit" means
            re-fetch, and each is reported in the frame's cache_status column so an operator can
            see what the cache is doing without watching the network.
        """
        if not entry:
            return "miss"
        if entry.get("fingerprint") != superevent_fingerprint(superevent):
            return "stale_fingerprint"
        if now is None:
            now = pd.Timestamp.now(tz="UTC")
        if pd.isna(gw_time) or (now - gw_time) < self.recheck_window:
            return "stale_age"
        return "hit"
