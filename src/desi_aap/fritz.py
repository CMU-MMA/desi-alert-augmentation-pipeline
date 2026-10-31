"""A small client for the Fritz (SkyPortal) API calls the pipeline makes.

Only the endpoints :mod:`desi_aap.stages.fritz_save` needs, as fritz.science
documents and implements them (SkyPortal, checked against the commit the site
reports at ``/api/sysinfo``):

``POST /api/brokers/{broker_id}/objects/{object_id}/save``
    Fetch an object from a broker (BOOM) and create its Obj, photometry and
    cutouts, saved to ``group_ids``. On an Obj that already exists it refreshes
    the photometry but does **not** add the object to the groups -- hence the
    next call.
``POST /api/sources``
    Save an object to groups. Idempotent: an existing source is left, an
    inactive one reactivated, a missing one created.
``GET /api/sources/{obj_id}``
    The source, including ``redshift`` and ``redshift_origin``.
``PATCH /api/sources/{obj_id}``
    Set ``redshift``, ``redshift_error``, ``redshift_origin``. There is no
    "only if unset" option, so the caller reads first.
``GET/POST /api/sources/{obj_id}/annotations``, ``PUT .../{annotation_id}``
    Annotations, unique per (object, origin). A PUT replaces ``data`` whole,
    and only the annotation's author may make one.

Every response is ``{"status": "success" | "error", "data": ..., "message": ...}``.
Authentication is the header ``Authorization: token <token>``. The token is
read from the environment by the caller and never logged or put in an error.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = ["FritzClient", "FritzError"]


class FritzError(RuntimeError):
    """A Fritz request failed: an HTTP error, or a response whose status is not success."""

    def __init__(self, method: str, path: str, status_code: int | None, message: str):
        self.method = method
        self.path = path
        self.status_code = status_code
        self.message = message
        code = f"HTTP {status_code}" if status_code is not None else "no response"
        super().__init__(f"{method} {path} failed ({code}): {message}")


class FritzClient:
    """Make Fritz API requests with one token, at a bounded rate.

    Parameters
    ----------
    base_url : str
        Such as ``"https://fritz.science"``.
    token : str
        A Fritz API token (Profile page). Kept only in the session headers.
    min_request_interval_s : float
        Smallest gap between two requests.
    timeout_s : float
        Per-request timeout.
    session : requests.Session, optional
        Injected by tests; a new one by default.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        min_request_interval_s: float = 0.25,
        timeout_s: float = 120.0,
        session: requests.Session | None = None,
    ):
        if not token:
            raise ValueError("A Fritz API token is required.")
        self.base_url = base_url.rstrip("/")
        self.min_request_interval_s = min_request_interval_s
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._session.headers.update({"Authorization": f"token {token}", "Accept": "application/json"})
        self._last_request = 0.0

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send one request and return its ``data``, raising FritzError on failure."""
        wait = self._last_request + self.min_request_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            response = self._session.request(
                method, f"{self.base_url}{path}", timeout=self.timeout_s, **kwargs
            )
        except requests.RequestException as exc:
            raise FritzError(method, path, None, type(exc).__name__) from None
        finally:
            self._last_request = time.monotonic()
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code >= 400 or not isinstance(body, dict) or body.get("status") != "success":
            message = body.get("message") if isinstance(body, dict) else None
            raise FritzError(method, path, response.status_code, str(message or response.reason or "")[:500])
        return body.get("data")

    def save_from_broker(self, broker_id: int, obj_id: str, group_ids: list[int]) -> Any:
        """Create the object from a broker, with photometry and cutouts, in ``group_ids``."""
        return self._request(
            "POST", f"/api/brokers/{broker_id}/objects/{obj_id}/save", json={"group_ids": list(group_ids)}
        )

    def save_source(
        self, obj_id: str, group_ids: list[int], *, ra: float | None = None, dec: float | None = None
    ) -> Any:
        """Save an object to ``group_ids``; ``ra``/``dec`` are needed only to create a new one."""
        body: dict[str, Any] = {"id": obj_id, "group_ids": list(group_ids)}
        if ra is not None and dec is not None:
            body.update(ra=float(ra), dec=float(dec))
        return self._request("POST", "/api/sources", json=body)

    def get_source(self, obj_id: str) -> dict[str, Any]:
        """The source as Fritz has it."""
        return self._request("GET", f"/api/sources/{obj_id}")

    def set_redshift(self, obj_id: str, redshift: float, error: float | None, origin: str) -> Any:
        """Set the source's redshift, its error (when known) and where it came from."""
        body: dict[str, Any] = {"redshift": float(redshift), "redshift_origin": origin}
        if error is not None:
            body["redshift_error"] = float(error)
        return self._request("PATCH", f"/api/sources/{obj_id}", json=body)

    def get_annotations(self, obj_id: str) -> list[dict[str, Any]]:
        """Every annotation on the source the token can read."""
        return self._request("GET", f"/api/sources/{obj_id}/annotations") or []

    def post_annotation(self, obj_id: str, origin: str, data: dict[str, Any], group_ids: list[int]) -> Any:
        """Add an annotation, visible to ``group_ids``."""
        return self._request(
            "POST",
            f"/api/sources/{obj_id}/annotations",
            json={"origin": origin, "data": data, "group_ids": list(group_ids)},
        )

    def put_annotation(
        self, obj_id: str, annotation_id: int, origin: str, data: dict[str, Any], group_ids: list[int]
    ) -> Any:
        """Replace an annotation's data."""
        return self._request(
            "PUT",
            f"/api/sources/{obj_id}/annotations/{annotation_id}",
            json={"origin": origin, "data": data, "group_ids": list(group_ids)},
        )
