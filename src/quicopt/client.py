# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.client — talk to the Quicopt service over HTTP.

Encode a model to wire bytes (``quicopt.wire``), POST it, read the result back.
Stdlib-only (``urllib``): the transport adds no dependency, like the rest of the
core. The request body is the versioned wire bytes; the response is the service's
result JSON (``status`` / ``objective`` / ``solution`` / a ready-to-print
``display`` / …). The first keyless call mints an API key, returned in the
``X-Quicopt-Api-Key`` response header and remembered on the :class:`Client` so
later calls replay it as ``Authorization: Bearer``.

Two entry points mirror the two service endpoints:

- :meth:`Client.solve` — POST ``/v1/solve``, block for the result (synchronous).
- :meth:`Client.submit` — POST ``/v1/jobs``, return a :class:`Job` to poll.

A non-2xx response raises :class:`QuicoptError`, carrying the service's stable
``reason`` code and the framed ``display`` text.
"""
from __future__ import annotations

import gzip as _gzip
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from urllib.parse import quote, urlencode

from .ir import Program
from .wire import encode

__all__ = ["Client", "Job", "Result", "QuicoptError", "DEFAULT_BASE_URL"]

_OCTET = "application/octet-stream"

DEFAULT_BASE_URL = "https://try.quicoptapi.pgi.fz-juelich.de"
"""The public Quicopt free-tier endpoint a :class:`Client` targets when no
``base_url`` is given. Mirrors the Julia client's ``DEFAULT_BASE_URL`` so both
clients reach the same server out of the box."""


@dataclass(frozen=True)
class Result:
    """A finished solve, parsed from the service's result JSON. ``objective`` and
    ``feasible`` are ``None`` when the class or outcome leaves them undefined
    (e.g. an unconstrained heuristic, or no incumbent). ``display`` is the framed,
    ready-to-print summary the service renders for every backend alike."""
    job_id: str
    status: str
    objective: Optional[float]
    feasible: Optional[bool]
    solution: Dict[str, float]
    solve_time_seconds: float
    solver_data: Dict[str, Any]
    display: str

    @property
    def model_class(self) -> Optional[str]:
        """The class the service routed the model to (``LP``/``MILP``/``QUBO``/…).

        Returns:
            The ``model_class`` recorded in ``solver_data``, or ``None`` if the
            service did not report one.
        """
        return self.solver_data.get("model_class")

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Result":
        """Build a ``Result`` from the service's decoded result JSON.

        Missing optional keys default rather than raise, so a partial result (e.g. a
        heuristic backend that reports no ``objective``) still parses.

        Args:
            d: The decoded result JSON object.

        Returns:
            Result: The parsed result.
        """
        return cls(
            job_id=d.get("job_id", ""),
            status=d["status"],
            objective=d.get("objective"),
            feasible=d.get("feasible"),
            solution=dict(d.get("solution") or {}),
            solve_time_seconds=d.get("solve_time_seconds", 0.0),
            solver_data=dict(d.get("solver_data") or {}),
            display=d.get("display", ""),
        )


class QuicoptError(Exception):
    """A non-2xx service response. ``reason`` is the service's stable snake_case
    code (``size_exceeded``, ``unsupported_model``, ``quota_exhausted``, …),
    ``display`` the framed message to print, ``status_code`` the HTTP status."""

    def __init__(self, status_code: int, body: Dict[str, Any]):
        self.status_code = status_code
        self.body = body if isinstance(body, dict) else {"error": str(body)}
        self.reason = self.body.get("reason")
        self.display = self.body.get("display")
        super().__init__(self.body.get("error") or f"HTTP {status_code}")


_Model = Any    # a front-end model (Pyomo, MathOpt, PuLP), a Program, or pre-encoded wire bytes


def _import_frontend(model: Any) -> Program:
    """A modeling-front-end object → ``Program``, via that front-end's importer,
    dispatched by type. The importers are imported lazily so the stdlib-only core
    never pulls in an optional front-end dependency until a model of that kind is
    actually solved."""
    mod = type(model).__module__ or ""
    if mod.startswith("ortools") or hasattr(model, "export_model"):
        from .mathopt import import_model
        return import_model(model)
    if mod.startswith("pyomo"):
        from .pyomo import import_model
        return import_model(model)
    if mod.startswith("pulp"):
        from .pulp import import_model
        return import_model(model)
    raise TypeError(f"cannot solve a {type(model).__name__}: pass a Pyomo, OR-Tools "
                    "MathOpt, or PuLP model, a Program, or wire bytes")


def _to_wire(model: _Model) -> bytes:
    """Coerce to wire bytes: pre-encoded bytes pass through, a ``Program`` is
    encoded, and a front-end model (Pyomo, MathOpt) is imported then encoded — so
    a caller solves the *model*, never a hand-built ``Program``."""
    if isinstance(model, (bytes, bytearray)):
        return bytes(model)
    if isinstance(model, Program):
        return encode(model)
    return encode(_import_frontend(model))


def _query(config: Optional[Dict[str, Any]]) -> str:
    """Render a config mapping as a URL query-string suffix.

    Encodes spaces as ``%20`` (``quote``, not the default ``+``) so a value round-
    trips unambiguously through the service's query parser, matching the Julia
    client.

    Args:
        config: Optional service parameters.

    Returns:
        str: ``"?k=v&…"`` if ``config`` is non-empty, else ``""``.
    """
    return "?" + urlencode(config, quote_via=quote) if config else ""


def _source_language(model: Any) -> Optional[str]:
    """The modelling front-end that authored ``model`` (for the ``source_language``
    tag), or ``None`` for a hand-built ``Program`` / pre-encoded wire bytes, which
    have no front-end to attribute. Mirrors the dispatch in ``_import_frontend``."""
    if isinstance(model, (bytes, bytearray, Program)):
        return None
    mod = type(model).__module__ or ""
    if mod.startswith("ortools") or hasattr(model, "export_model"):
        return "mathopt"
    if mod.startswith("pyomo"):
        return "pyomo"
    if mod.startswith("pulp"):
        return "pulp"
    return None


def _meta_config(model: Any, project: Optional[str],
                 config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the per-call metadata tags into the request config (sent as query
    params): the auto-detected ``source_language`` (unless the caller already set
    one in ``config``) and the optional ``project_id``. The model is unchanged —
    these ride the query string, not the wire bytes."""
    meta: Dict[str, Any] = dict(config or {})
    src = _source_language(model)
    if src is not None:
        meta.setdefault("source_language", src)
    if project is not None:
        meta["project_id"] = project
    return meta


class Client:
    """A connection to a Quicopt service at ``base_url`` — the public free tier
    (:data:`DEFAULT_BASE_URL`) unless another URL is given. Holds the API key: pass
    a known one, or let the first keyless call mint one (then read it back from
    ``client.api_key`` to persist)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: Optional[str] = None,
                 *, timeout: float = 60.0):
        """Bind a client to a service endpoint.

        Args:
            base_url: The service base URL; a trailing slash is stripped. Defaults
                to :data:`DEFAULT_BASE_URL`, the public free-tier endpoint; pass
                another URL to target a different server.
            api_key: A known API key, or ``None`` to let the first keyless call mint
                one (afterwards readable back from ``self.api_key``).
            timeout: Per-request socket timeout, in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def solve(self, model: _Model, *, project: Optional[str] = None,
              config: Optional[Dict[str, Any]] = None, gzip: bool = False) -> Result:
        """Solve ``model`` synchronously, blocking until the result returns.

        Args:
            model: A front-end model (Pyomo, OR-Tools MathOpt) — imported to the
                wire IR here — or a ``Program`` / pre-encoded wire bytes if you built
                them yourself.
            project: Optional project tag for the call, so calls on one key can be
                invoiced per project. Sent as a query param, not baked into the model.
            config: Optional service parameters, sent as the query string.
            gzip: If ``True``, gzip-compress the request body.

        Returns:
            Result: The finished solve.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        return Result._from_json(
            self._request("POST", "/v1/solve", _to_wire(model),
                          config=_meta_config(model, project, config), gzip=gzip))

    def submit(self, model: _Model, *, project: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None, gzip: bool = False) -> "Job":
        """Submit ``model`` for asynchronous solving and return a handle to poll.

        Args:
            model: A Pyomo/MathOpt model, a ``Program``, or pre-encoded wire bytes.
            project: Optional project tag for the call (per-project invoicing), sent
                as a query param, not baked into the model.
            config: Optional service parameters, sent as the query string.
            gzip: If ``True``, gzip-compress the request body.

        Returns:
            Job: A handle to the queued job; call :meth:`Job.result` to await it.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        body = self._request("POST", "/v1/jobs", _to_wire(model),
                             config=_meta_config(model, project, config), gzip=gzip)
        return Job(self, body["job_id"])

    # ── transport ───────────────────────────────────────────────────────────

    def _open(self, method: str, path: str, data: Optional[bytes] = None, *,
              config: Optional[Dict[str, Any]] = None, gzip: bool = False) -> bytes:
        """Perform one HTTP request and return the raw response body.

        Sets the octet-stream content type (and gzip encoding, if requested) for a
        body, attaches the bearer key when known, and captures a freshly minted key
        off the response — including from an ``HTTPError``, since a 502/504 on a
        first call may still have minted one.

        Args:
            method: The HTTP method.
            path: The path under ``base_url`` (e.g. ``/v1/solve``).
            data: The request body, or ``None`` for a bodyless request.
            config: Optional service parameters, sent as the query string.
            gzip: If ``True``, gzip-compress ``data``.

        Returns:
            bytes: The raw response body.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        headers: Dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = _OCTET
            if gzip:
                data = _gzip.compress(data)
                headers["Content-Encoding"] = "gzip"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base_url + path + _query(config),
                                     data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self._capture_key(resp.headers)
                return resp.read()
        except urllib.error.HTTPError as e:
            self._capture_key(e.headers)        # a 504/502 on a first call still minted a key
            raise QuicoptError(e.code, _decode(e.read())) from None

    def _request(self, method: str, path: str, data: Optional[bytes] = None, **kw) -> Dict[str, Any]:
        """Perform a request via :meth:`_open` and parse its JSON body.

        Args:
            method: The HTTP method.
            path: The path under ``base_url``.
            data: The request body, or ``None``.
            **kw: Forwarded to :meth:`_open` (``config``, ``gzip``).

        Returns:
            dict: The parsed JSON object, or ``{}`` for an empty body.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        raw = self._open(method, path, data, **kw)
        return json.loads(raw) if raw else {}

    def _capture_key(self, headers) -> None:
        """Remember a freshly minted API key from a response, if we have none yet.

        The first keyless call mints a key, returned in the ``X-Quicopt-Api-Key``
        response header; storing it lets the next call authenticate as the same
        caller. A key we already hold is never overwritten.

        Args:
            headers: The response headers.

        Returns:
            None. ``self.api_key`` is set on first mint.
        """
        minted = headers.get("X-Quicopt-Api-Key")
        if minted and not self.api_key:
            self.api_key = minted


def _decode(raw: bytes) -> Dict[str, Any]:
    """Best-effort decode of an error response body into a dict.

    Args:
        raw: The raw response body (from a non-2xx response).

    Returns:
        dict: The parsed JSON if it parses (or ``{}`` when empty), else the body
        wrapped as ``{"error": <text>}`` so a non-JSON error still surfaces.
    """
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"error": raw.decode("utf-8", "replace")}


@dataclass
class Job:
    """A handle to an async job. :meth:`result` polls until it finishes."""
    client: Client
    job_id: str

    def status(self) -> Dict[str, Any]:
        """Fetch the job's metadata and framed ``log_tail``.

        Returns:
            dict: The job state (``queued``/``running``/``done``/``failed``) and its
            log tail, as returned by the service.
        """
        return self.client._request("GET", f"/v1/jobs/{self.job_id}")

    def result(self, *, wait: bool = True, timeout: float = 120.0, poll: float = 0.5) -> Result:
        """Fetch the job's result, optionally polling until it is ready.

        Args:
            wait: If ``True``, poll past the service's ``not_done`` reason until the
                worker finishes; if ``False``, fetch once.
            timeout: Maximum time to poll, in seconds, before giving up.
            poll: Delay between polls, in seconds.

        Returns:
            Result: The finished solve.

        Raises:
            QuicoptError: If ``wait`` is ``False`` and the job is not yet done, on
                any non-``not_done`` error, or once ``timeout`` elapses.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return Result._from_json(
                    self.client._request("GET", f"/v1/jobs/{self.job_id}/result"))
            except QuicoptError as e:
                if not wait or e.reason != "not_done" or time.monotonic() > deadline:
                    raise
                time.sleep(poll)

    def log(self) -> str:
        """Fetch the job's plain-text log.

        Returns:
            str: The framed log view on success, or the error text on failure.
        """
        return self.client._open("GET", f"/v1/jobs/{self.job_id}/log").decode("utf-8", "replace")

    def delete(self) -> None:
        """Delete the job and its stored blob/result/log on the server.

        Returns:
            None.
        """
        self.client._open("DELETE", f"/v1/jobs/{self.job_id}")
