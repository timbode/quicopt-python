# quicopt

The Python client for the Quicopt optimization service. Author a model in a Python
modeling front-end (Pyomo, or OR-Tools MathOpt), convert it to Quicopt's wire IR, and
emit the versioned, language-neutral bytes the service consumes.

## Install

The conventional Python workflow is an editable install into a virtual environment:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # core (ir + wire) — standard library only
pip install -e '.[pyomo]'   # + the Pyomo front-end
pip install -e '.[mathopt]' # + the OR-Tools MathOpt front-end
```

## Use

```python
import pyomo.environ as pyo
from quicopt import Client

m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0.1, 10))
m.obj = pyo.Objective(expr=m.x**2 + 1.0 / m.x, sense=pyo.minimize)

client = Client("https://quicopt.example")   # your service endpoint
result = client.solve(m)                      # solve the model — the import to the
                                              # wire IR happens inside
print(result.status, result.objective, result.solution)
print(result.display)                         # the service's ready-to-print summary
```

`solve` takes the model directly (Pyomo, or an OR-Tools MathOpt model) and imports it
to the wire IR internally. The first keyless call mints an API key (`client.api_key`);
reuse it on later calls (`Client(url, api_key=…)`). For a long solve, `client.submit(m)`
returns a job handle to poll — `job.result()`.

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
quicopt/client.py  POST the wire bytes to the service, read the result (HTTP, stdlib)
```

The IR and wire format are the client's contract with the service; `wire.py`
encodes that schema exactly. Each front-end is an independent module beside
`pyomo.py` (`mathopt.py` for OR-Tools authors; further modeling libraries slot in
the same way) and pulls in only its own optional extra.

## Test

The encoder is checked against committed golden byte vectors, with **no
dependencies**:

```sh
python3 tests/test_wire_golden.py        # or: pytest tests/
```

## Status

- **ir + wire** — stable; the encoder is byte-exact against what the service decodes.
- **pyomo importer** — affine / quadratic / nonlinear (`+ - * / ^ sin cos exp log
  sqrt abs`), variable bounds (incl. unbounded) + integrality, `==` / `<=` / `>=` /
  ranged constraints, `min` / `max`.
- **mathopt importer** — OR-Tools MathOpt `ModelProto`: linear / quadratic
  objective, linear constraints (incl. ranged and one-sided), variable bounds
  (incl. unbounded) + integrality, `min` / `max`.
- **transport (HTTP)** — `Client.solve` / `Client.submit` over `/v1/solve` and
  `/v1/jobs`: wire bytes up, result JSON (status / objective / solution / framed
  `display`) back; API-key minting on the first call, optional gzip. Standard
  library only.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich.
