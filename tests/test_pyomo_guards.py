# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
The Pyomo importer's guard rails — the cases that must raise rather than mislead.

An import that fails loudly costs the author a fix; one that quietly changes the
model's meaning costs them a wrong answer they have no reason to doubt. These tests
pin the second kind down.

Skips unless the ``[pyomo]`` extra is installed:
    pip install -e '.[pyomo]'
Run with:
    python3 tests/test_pyomo_guards.py       # or: pytest tests/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

try:
    import pyomo.environ as pyo
    from quicopt.pyomo import import_model
except ImportError:                                         # the extra is absent
    pyo = None


def test_fixed_var_without_value_raises():
    """A variable fixed with no value must raise, not silently become free.

    ``m.x.fix()`` with no argument leaves ``fixed=True, value=None``. Reading an
    absent bound as ±Inf — right for an unbounded variable — would then turn the pin
    into a free variable, and the model would solve to a confidently wrong answer.
    """
    if pyo is None:
        print("skip  test_fixed_var_without_value_raises (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.y = pyo.Var()
    m.y.fix()                                               # fixed, but to nothing
    m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
    try:
        import_model(m)
        assert False, "expected ValueError — a fixed var with no value must not import"
    except ValueError as e:
        assert "y" in str(e), f"the error must name the offending variable, got: {e}"


def test_fixed_var_with_value_still_pins():
    """The guard must not catch the ordinary case: a fixed var pins to [val, val]."""
    if pyo is None:
        print("skip  test_fixed_var_with_value_still_pins (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.y = pyo.Var()
    m.y.fix(2.5)
    m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
    y = import_model(m).vars[1]
    assert (y.lower, y.upper, y.start) == (2.5, 2.5, 2.5), f"fixed var not pinned: {y}"


def test_free_var_still_unbounded():
    """The guard must not catch an ordinary *unfixed* var with no bounds either."""
    if pyo is None:
        print("skip  test_free_var_still_unbounded (needs the [pyomo] extra)")
        return
    from math import inf

    m = pyo.ConcreteModel()
    m.x = pyo.Var()                                         # free in ℝ — absent bounds are ±Inf
    m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
    x = import_model(m).vars[0]
    assert (x.lower, x.upper) == (-inf, inf), f"free var not unbounded: {x}"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all pyomo guard tests passed")
