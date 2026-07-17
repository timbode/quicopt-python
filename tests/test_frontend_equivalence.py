# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
Cross-front-end byte equality: the same model, authored twice, is the same wire.

``quicopt.pyomo`` is pinned to the service's codec by the goldens
(``test_wire_golden.py``). This test pins ``quicopt.pulp`` to *it*: the same MILP is
written once in Pyomo and once in PuLP, and the two front-ends must produce
identical bytes. Identical bytes ⇒ an identical decoded ``Program``, so a match
carries the golden's authority across to the second front-end — with no reference
codec, no network, and no solver.

The model is deliberately offset-free in the objective. The front-ends order an
objective constant differently (Pyomo emits it last, following its own expression
tree; PuLP reads a coefficient table and emits it first), which is a structural
difference the service is indifferent to but byte-equality is not.

Skips unless both optional extras are installed:
    pip install -e '.[pyomo,pulp]'
Run with:
    python3 tests/test_frontend_equivalence.py    # or: pytest tests/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

from quicopt import encode                                  # noqa: E402

try:
    import pyomo.environ as pyo
    import pulp
    from quicopt.pyomo import import_model as from_pyomo
    from quicopt.pulp import import_model as from_pulp
except ImportError:                                         # an extra is absent — nothing to compare
    pyo = None


# The model, twice —  max 3x₁ + 2x₂  s.t.  2x₁ + x₂ ≤ 5,  x₁ − x₂ = 1,
#                     x₁ ∈ [0, 4] ⊂ ℝ,  x₂ ∈ {0, 1}.
# Pyomo names variables positionally (x1, x2, … in declaration order), PuLP by the
# author's name, so the PuLP names are chosen to agree.

def _pyomo_model():
    """Build the reference MILP as a Pyomo ``ConcreteModel``.

    Returns:
        ConcreteModel: The model, with variables declared in ``x1``, ``x2`` order.
    """
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, 4))
    m.x2 = pyo.Var(domain=pyo.Binary)
    m.obj = pyo.Objective(expr=3 * m.x1 + 2 * m.x2, sense=pyo.maximize)
    m.c1 = pyo.Constraint(expr=2 * m.x1 + m.x2 <= 5)
    m.c2 = pyo.Constraint(expr=m.x1 - m.x2 == 1)
    return m


def _pulp_model():
    """Build the same MILP as a PuLP ``LpProblem``.

    Returns:
        LpProblem: The model, its variables named to match the Pyomo declaration order.
    """
    x1 = pulp.LpVariable("x1", 0, 4)
    x2 = pulp.LpVariable("x2", cat=pulp.LpBinary)
    p = pulp.LpProblem("m", pulp.LpMaximize)
    p += 3 * x1 + 2 * x2
    p += 2 * x1 + x2 <= 5
    p += x1 - x2 == 1
    return p


def test_pulp_matches_pyomo_bytes():
    """The PuLP and Pyomo front-ends encode the same model to the same bytes."""
    if pyo is None:
        print("skip  test_pulp_matches_pyomo_bytes (needs the [pyomo,pulp] extras)")
        return
    assert encode(from_pulp(_pulp_model())) == encode(from_pyomo(_pyomo_model()))


def test_feasibility_problem_has_constant_objective():
    """A PuLP problem with no objective is a feasibility problem: a constant 0."""
    if pyo is None:
        print("skip  test_feasibility_problem_has_constant_objective (needs the extras)")
        return
    from quicopt import Const

    x = pulp.LpVariable("x", 0, 4)
    p = pulp.LpProblem("feas", pulp.LpMinimize)
    p += 2 * x >= 3
    prog = from_pulp(p)
    assert prog.objective == Const(0.0) and prog.sense == "min"
    assert len(prog.constraints) == 1
    encode(prog)                                            # the constant objective is encodable


def test_duplicate_names_raise():
    """Two variables sharing a name would merge into one — that must raise."""
    if pyo is None:
        print("skip  test_duplicate_names_raise (needs the extras)")
        return
    a, b = pulp.LpVariable("a b"), pulp.LpVariable("a-b")   # PuLP sanitizes both to "a_b"
    p = pulp.LpProblem("dup", pulp.LpMinimize)
    p += a + b
    p += a + b >= 1
    try:
        from_pulp(p)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not unique" in str(e)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all front-end equivalence tests passed")
