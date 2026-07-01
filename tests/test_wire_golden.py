# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
The byte-equality test — the encoder's correctness property.

For each fixture, ``quicopt.wire.encode`` must reproduce, byte for byte, the
committed golden bytes the Quicopt service expects (``tests/goldens/<name>.hex``).
Byte-equality is the sharpest check: identical bytes ⇒ identical decoded ``Program``
(the codec is injective), so a match proves structural fidelity. The goldens are
committed fixtures; the service's own codec is what they were captured from.

Run with no dependencies installed:
    python3 tests/test_wire_golden.py
or under pytest:
    pytest tests/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

from quicopt import (Const, Param, Var, Apply, Reduce, SetRef,         # noqa: E402
                     Zero, Nonneg, Indicator,
                     VarDecl, IndexSet, Constraint, Program,
                     CONTINUOUS, BINARY, encode)

GOLD = ROOT / "tests" / "goldens"


# Fixtures — the same Programs the service's codec encodes for the goldens.

def _scalar_nlp():                                          # min x1² + 1/x1, x1 ∈ [0.1, 10]
    x = Var("x1")
    return Program(
        vars=[VarDecl("x1", [], CONTINUOUS, 0.1, 10.0, 0.1)],
        objective=Apply("+", [Apply("^", [x, Const(2.0)]), Apply("/", [Const(1.0), x])]),
        sense="min")


def _bounds_zero_start():                                   # start=0, fix=0, max, Nonneg+Zero
    y = Var("y1")
    return Program(
        vars=[VarDecl("y1", [], CONTINUOUS, -1.0, 1.0, 0.0)],
        objective=y, sense="max",
        constraints=[Constraint(y, Nonneg(), []),
                     Constraint(Apply("-", [y, Const(0.5)]), Zero(), [])],
        fix={("y1", ()): 0.0})


def _indexed():                                             # families, indexing, canonical sort
    # Names are deliberately abstract: this fixture exercises the encoder's handling
    # of index sets, dependent sets, param tables and reductions — not any domain.
    sets = [IndexSet("S", [1, 2, 3])]
    indexed = {"a": {(1,): [10, 11], (3,): [30], (2,): [20]}}
    params = {"p": {(2,): 2.0, (1,): 0.0, (3,): 3.0}}
    vars = [
        VarDecl("α", ["S"], CONTINUOUS, -3.15, 3.15, 0.0),
        VarDecl("y", ["S"], CONTINUOUS, "p", 10.0, 1.0),        # lower = Param name ⇒ Bound::param
        VarDecl("z", ["S"], BINARY, 0.0, 1.0, 0.0)]
    obj = Reduce("+", "i", SetRef("S", ()),
                 Apply("*", [Param("p", ("i",)), Var("y", ("i",))]), None)
    cons = [Constraint(
        Reduce("+", "j", SetRef("a", ("i",)), Var("y", ("j",)), Var("z", ("j",))),
        Nonneg(),
        [("i", SetRef("S", ()))])]
    fix = {("α", (1,)): 0.0}
    return Program(sets, indexed, params, vars, obj, "min", cons, fix)


def _indicator():                                           # u ⟹ x ≥ 0
    x = Var("x")
    u = Var("u")
    return Program(
        vars=[VarDecl("x", [], CONTINUOUS, -5.0, 5.0, 0.0),
              VarDecl("u", [], BINARY, 0.0, 1.0, 0.0)],
        objective=x, sense="min",
        constraints=[Constraint(x, Indicator(u, Nonneg()), [])])


FIXTURES = {
    "scalar_nlp": _scalar_nlp,
    "bounds_zero_start": _bounds_zero_start,
    "indexed": _indexed,
    "indicator": _indicator,
}


def _golden(name):
    return bytes.fromhex((GOLD / f"{name}.hex").read_text().strip())


def test_goldens():
    for name, build in FIXTURES.items():
        got, want = encode(build()), _golden(name)
        assert got == want, f"{name}: {got.hex()} != {want.hex()}"


if __name__ == "__main__":
    ok = True
    for name, build in FIXTURES.items():
        got, want = encode(build()), _golden(name)
        if got == want:
            print(f"PASS  {name:<20} {len(got):>4} bytes")
        else:
            ok = False
            print(f"FAIL  {name}\n   got  {got.hex()}\n   want {want.hex()}")
    sys.exit(0 if ok else 1)
