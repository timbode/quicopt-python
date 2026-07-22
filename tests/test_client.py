# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
Transport tests, self-contained: a stdlib HTTP stub stands in for the Quicopt
service, so the client's request shaping (wire body, gzip, bearer auth) and
response parsing (result JSON, minted key, error → exception, polling) are
exercised with no network, no service, and no test dependency beyond the standard
library.

    python3 tests/test_client.py        # or: pytest tests/test_client.py
"""
import contextlib
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from quicopt import (Apply, Client, Const, Constraint, CONTINUOUS, DEFAULT_BASE_URL,
                     Job, Nonneg, Program, QuicoptError, Var, VarDecl, encode)

_KEY = "k" * 64
_JOB = "job-1"
_RESULT = {"job_id": _JOB, "status": "optimal", "objective": 4.0, "feasible": True,
           "solve_time_seconds": 0.01, "solution": {"x0": 4.0},
           "solver_data": {"model_class": "lp", "n_variables": 1},
           "display": "\n  Quicopt · optimal\n  └── objective 4.0"}
_ERROR = {"error": "model not supported in the free tier (nonlinear program)",
          "reason": "unsupported_model",
          "display": "\n  Quicopt · Not supported on the free tier yet"}


class _Stub(BaseHTTPRequestHandler):
    def log_message(self, *_):                       # keep the test output quiet
        pass

    def _json(self, code, obj, extra=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        return gzip.decompress(data) if self.headers.get("Content-Encoding") == "gzip" else data

    def do_POST(self):
        srv = self.server
        srv.last_body = self._read()
        srv.last_auth = self.headers.get("Authorization")
        srv.last_encoding = self.headers.get("Content-Encoding")
        srv.last_path = self.path
        minted = {} if self.headers.get("Authorization") else {"X-Quicopt-Api-Key": _KEY}
        if srv.mode == "error":
            self._json(422, _ERROR, minted)
        elif self.path.startswith("/v1/solve"):
            self._json(200, _RESULT, minted)
        elif self.path.startswith("/v1/jobs"):
            self._json(202, {"job_id": _JOB, "status": "queued"}, minted)
        else:
            self._json(404, {"error": "no route"})

    def do_GET(self):
        srv = self.server
        if self.path.endswith("/result"):
            if srv.not_done_first and srv.gets == 0:    # one not-yet-done, then ready
                srv.gets += 1
                self._json(409, {"reason": "not_done", "error": "job not done"})
            else:
                self._json(200, _RESULT)
        elif "/jobs/" in self.path:
            self._json(200, {"job_id": _JOB, "status": "done"})
        else:
            self._json(404, {"error": "no route"})


@contextlib.contextmanager
def _serve(mode="ok", not_done_first=False):
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    srv.mode, srv.not_done_first, srv.gets = mode, not_done_first, 0
    srv.last_body = srv.last_auth = srv.last_encoding = srv.last_path = None
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield Client(f"http://127.0.0.1:{srv.server_address[1]}", timeout=5.0), srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join()


def _program():
    # max x0 s.t. x0 ≤ 4, 0 ≤ x0 ≤ 10 — a scalar LP, so the test sends real wire bytes
    x = Var("x0")
    return Program(vars=[VarDecl("x0", [], CONTINUOUS, 0.0, 10.0, 0.0)],
                   objective=x, sense="max",
                   constraints=[Constraint(Apply("-", [Const(4.0), x]), Nonneg())])


def test_solve_roundtrip_and_key_mint():
    prog = _program()
    with _serve() as (client, srv):
        res = client.solve(prog)
        assert res.status == "optimal" and res.objective == 4.0
        assert res.solution == {"x0": 4.0} and res.model_class == "lp"
        assert srv.last_body == encode(prog)            # the Program was encoded and sent
        assert srv.last_auth is None                     # first call is keyless
        assert client.api_key == _KEY                    # minted key captured off the header
        client.solve(prog)
        assert srv.last_auth == "Bearer " + _KEY         # replayed as bearer on the next call


def test_metadata_source_and_project():
    from quicopt.client import _source_language, _meta_config

    def fake(mod, **attrs):
        cls = type("M", (), attrs)
        cls.__module__ = mod
        return cls()

    # source auto-detection by module / duck-type; hand-built inputs have none
    assert _source_language(fake("pyomo.environ")) == "pyomo"
    assert _source_language(fake("pulp.pulp")) == "pulp"
    assert _source_language(fake("ortools.math_opt.python.model")) == "mathopt"
    assert _source_language(fake("whatever", export_model=lambda self: None)) == "mathopt"
    assert _source_language(_program()) is None
    assert _source_language(b"\x00") is None

    # _meta_config merges auto source (unless overridden) + project
    assert _meta_config(_program(), "proj-1", None) == {"project_id": "proj-1"}
    assert _meta_config(fake("pyomo.environ"), None, None) == {"source_language": "pyomo"}
    assert _meta_config(fake("pyomo.environ"), None, {"source_language": "x"})["source_language"] == "x"

    # end-to-end: project rides the request query string (%-escaped), not the body
    prog = _program()
    with _serve() as (client, srv):
        client.solve(prog, project="proj A/1")
        assert "project_id=proj%20A%2F1" in srv.last_path
        assert srv.last_body == encode(prog)


def test_default_base_url():
    # Omitting base_url targets the public free tier; an explicit URL still wins.
    assert Client().base_url == DEFAULT_BASE_URL == "https://try.quicoptapi.pgi.fz-juelich.de"
    assert Client("http://127.0.0.1:9").base_url == "http://127.0.0.1:9"


def test_solve_gzip():
    prog = _program()
    with _serve() as (client, srv):
        client.solve(prog, gzip=True)
        assert srv.last_encoding == "gzip"
        assert srv.last_body == encode(prog)            # the service sees it decompressed


def test_solve_accepts_raw_bytes():
    with _serve() as (client, srv):
        client.solve(b"\x01\x02\x03")                    # pre-encoded wire bytes pass through
        assert srv.last_body == b"\x01\x02\x03"


def test_solve_rejects_unknown_model():
    # A front-end model imports inside solve(); anything that is neither a known
    # front-end model, a Program, nor wire bytes is a clear TypeError (not a wire
    # request). The positive Pyomo/MathOpt/PuLP dispatch is covered where those
    # extras are installed — it stays out of this stdlib-only suite.
    with _serve() as (client, srv):
        try:
            client.solve(object())
        except TypeError as e:
            assert "Pyomo" in str(e) and "MathOpt" in str(e) and "PuLP" in str(e)
        else:
            assert False, "expected TypeError for an unknown model type"
        assert srv.last_body is None                      # nothing was sent


def test_error_response_raises():
    with _serve(mode="error") as (client, _):
        try:
            client.solve(_program())
        except QuicoptError as e:
            assert e.status_code == 422
            assert e.reason == "unsupported_model"
            assert e.display and "Quicopt" in e.display
        else:
            assert False, "expected QuicoptError"


def test_submit_then_result():
    prog = _program()
    with _serve() as (client, _):
        job = client.submit(prog)
        assert job.job_id == _JOB
        res = job.result()
        assert res.status == "optimal" and res.objective == 4.0


def test_result_polls_past_not_done():
    with _serve(not_done_first=True) as (client, srv):
        res = Job(client, _JOB).result(poll=0.01)
        assert res.objective == 4.0
        assert srv.gets == 1                             # one 409 not_done, then a 200


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all client transport tests passed")
