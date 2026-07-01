# quicopt

The Python client for the Quicopt optimization service. Author a model in a Python
modeling front-end ([Pyomo](https://www.pyomo.org/), or
[OR-Tools MathOpt](https://developers.google.com/optimization/math_opt)), convert it
to Quicopt's wire IR, and emit the versioned, language-neutral bytes the service
consumes.

The core (`ir` + `wire`) depends on nothing outside the standard library; each
front-end is an optional extra.

## Install

```sh
pip install quicopt              # core (ir + wire) — standard library only
pip install "quicopt[pyomo]"     # + the Pyomo front-end
pip install "quicopt[mathopt]"   # + the OR-Tools MathOpt front-end
```

## Quickstart

```python
import pyomo.environ as pyo
from quicopt import Client

m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0.1, 10))
m.obj = pyo.Objective(expr=m.x**2 + 1.0 / m.x, sense=pyo.minimize)

client = Client("https://quicopt.example")   # your service endpoint
result = client.solve(m)                      # the import to the wire IR happens inside
print(result.status, result.objective, result.solution)
print(result.display)                         # the service's ready-to-print summary
```

[`solve`][quicopt.client.Client.solve] takes the model directly (Pyomo, or an
OR-Tools MathOpt model) and imports it to the wire IR internally. For a long solve,
[`submit`][quicopt.client.Client.submit] returns a [`Job`][quicopt.client.Job] handle
to poll.

If you need the wire bytes yourself — to inspect them or send them by another route —
build a [`Program`][quicopt.ir.Program] (directly or via a front-end
`import_model`) and call [`encode`][quicopt.wire.encode].

## How it fits together

```
front-end model ──import_model──▶ Program (ir) ──encode──▶ wire bytes ──▶ service
   (Pyomo,                         the language-neutral    the versioned
    MathOpt)                       contract                on-the-wire form
```

See the **API reference** for each layer:

- [`ir`](api/ir.md) — the `Program` IR data model (front-end-agnostic).
- [`wire`](api/wire.md) — `Program` → versioned wire bytes.
- [`pyomo`](api/pyomo.md) — a Pyomo model → `Program`.
- [`mathopt`](api/mathopt.md) — an OR-Tools MathOpt model → `Program`.
- [`client`](api/client.md) — POST the wire bytes and read the result back.
