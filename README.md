# quicopt

[![PyPI](https://img.shields.io/pypi/v/quicopt.svg)](https://pypi.org/project/quicopt/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/quicopt/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://timbode.github.io/quicopt-python/)
[![Downloads](https://img.shields.io/pypi/dm/quicopt.svg)](https://pypistats.org/packages/quicopt)
[![License: Apache 2.0](https://img.shields.io/pypi/l/quicopt.svg)](https://github.com/timbode/quicopt-python/blob/main/LICENSE)

The Python client for the Quicopt optimization service. Author a model in a Python
modeling front-end (Pyomo, OR-Tools MathOpt, or PuLP), convert it to Quicopt's wire
IR, and emit the versioned, language-neutral bytes the service consumes.

## Install

```sh
pip install quicopt              # core (ir + wire) — standard library only
pip install "quicopt[pyomo]"     # + the Pyomo front-end
pip install "quicopt[mathopt]"   # + the OR-Tools MathOpt front-end
pip install "quicopt[pulp]"      # + the PuLP front-end
```

From source (contributors), an editable install into a virtual environment:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[pyomo,mathopt,pulp]'
```

## Use

```python
import pyomo.environ as pyo
from quicopt import Client

m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0.1, 10))
m.obj = pyo.Objective(expr=m.x**2 + 1.0 / m.x, sense=pyo.minimize)

client = Client()                             # defaults to the free-tier Quicopt server
result = client.solve(m)                      # solve the model — the import to the
                                              # wire IR happens inside
print(result.status, result.objective, result.solution)
print(result.display)                         # the service's ready-to-print summary
```

`solve` takes the model directly (Pyomo, OR-Tools MathOpt, or PuLP) and imports it
to the wire IR internally. The first keyless call mints an API key (`client.api_key`);
reuse it on later calls (`Client(api_key=…)`). Point `Client(base_url=…)` at another
server to override the default. For a long solve, `client.submit(m)` returns a job
handle to poll — `job.result()`.

Tag a call with `client.solve(m, project="my-project")` to attribute it to a
project — handy when one key serves several projects. The modelling front-end
(Pyomo/MathOpt/PuLP) is recorded automatically.

If you need the wire bytes yourself (to inspect or send by another route), the
front-end importers and encoder are still public:

```python
from quicopt import encode
from quicopt.pyomo import import_model

payload = encode(import_model(m))   # Pyomo model → Program → versioned wire bytes
```

## Layout

```
quicopt/ir.py      the Program IR data model
quicopt/wire.py    Program → versioned wire bytes (a stdlib-only encoder)
quicopt/pyomo.py   Pyomo model → Program (a front-end)
quicopt/mathopt.py OR-Tools MathOpt model → Program (a front-end)
quicopt/pulp.py    PuLP model → Program (a front-end)
quicopt/client.py  POST the wire bytes to the service, read the result (HTTP, stdlib)
```

The IR and wire format are the client's contract with the service; `wire.py`
encodes that schema exactly. Each front-end is an independent module beside
`pyomo.py` (`mathopt.py` for OR-Tools authors, `pulp.py` for PuLP authors; further
modeling libraries slot in the same way), pulls in only its own optional extra, and
builds the IR through the canonical forms in `_terms.py` — so the same model reaches
the same wire whichever front-end authored it.

## Test

The encoder is checked against committed golden byte vectors, with **no
dependencies**:

```sh
python3 tests/test_wire_golden.py        # or: pytest tests/
```

The front-ends are pinned to each other by byte equality: the same model authored in
Pyomo and in PuLP must encode to identical bytes, which carries the golden's
authority across (`tests/test_frontend_equivalence.py`; needs the `[pyomo,pulp]`
extras, skips without them).

## Status

- **ir + wire** — stable; the encoder is byte-exact against what the service decodes.
- **pyomo importer** — affine / quadratic / nonlinear (`+ - * / ^ sin cos exp log
  sqrt abs`), variable bounds (incl. unbounded) + integrality, `==` / `<=` / `>=` /
  ranged constraints, `min` / `max`. A fixed variable pins to `[val, val]`; one fixed
  *without* a value raises rather than importing as free.
- **mathopt importer** — OR-Tools MathOpt `ModelProto`: linear / quadratic
  objective, linear constraints (incl. ranged and one-sided), variable bounds
  (incl. unbounded) + integrality, `min` / `max`.
- **pulp importer** — PuLP `LpProblem`: linear objective (with offset) and linear
  `<=` / `==` / `>=` constraints, variable bounds (incl. unbounded) + integrality,
  `min` / `max`. PuLP is linear by construction, so this is exactly LP / MILP; a
  problem with no objective is a feasibility problem (a constant `0`).
- **transport (HTTP)** — `Client.solve` / `Client.submit` over `/v1/solve` and
  `/v1/jobs`: wire bytes up, result JSON (status / objective / solution / framed
  `display`) back; API-key minting on the first call, optional gzip. Standard
  library only.

## License

Apache License 2.0 — see [`LICENSE`](https://github.com/timbode/quicopt-python/blob/main/LICENSE). (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich.
