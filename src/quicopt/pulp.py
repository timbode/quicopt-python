# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.pulp — a PuLP problem → Quicopt ``Program`` IR.

A third front-end (beside ``quicopt.pyomo`` and ``quicopt.mathopt``), for authors
already holding a `PuLP <https://coin-or.github.io/pulp/>`_ model. There is no
expression tree to walk: a PuLP ``LpAffineExpression`` *is* a mapping from variable
to coefficient, so the objective ``offset + Σ cᵢxᵢ`` and each constraint
``Σ cᵢxᵢ + k ⋈ 0`` are read off directly, with variable bounds + category into
``VarDecl`` and the constraint rows into ``Zero``/``Nonneg``. An absent PuLP bound
(``None``) is ±Inf — a free direction.

PuLP is linear by construction, so the supported surface is exactly LP / MILP: it
has no quadratic expression type, hence no QUBO to import (unlike ``quicopt.mathopt``,
which can express one). Everything a PuLP model can say, the IR serves.

Requires the optional ``[pulp]`` extra (``pip install -e '.[pulp]'``).
"""
from __future__ import annotations

from math import inf

import pulp

from ._terms import _emit, _scaled, _sum
from .ir import Const, Program, Var, VarDecl, BINARY, CONTINUOUS, INTEGER

# PuLP writes every constraint as `f + k ⋈ 0`, so the bound is b = −k. Each sense
# maps to the range `lb ≤ f ≤ ub` the shared emitter takes; an open side is ±Inf.
_RANGE = {
    pulp.LpConstraintEQ: lambda b: (b, b),
    pulp.LpConstraintGE: lambda b: (b, inf),
    pulp.LpConstraintLE: lambda b: (-inf, b),
}


def _affine(expr):
    """Read a PuLP ``LpAffineExpression``'s variable terms as an IR expression.

    The expression's own ``constant`` is *not* included — the objective folds it in
    as an offset, a constraint carries it as the negated bound.

    Args:
        expr: The PuLP ``LpAffineExpression`` (a variable → coefficient mapping).

    Returns:
        Expression: ``Σ cᵢxᵢ`` as an IR expression, ``Const(0.0)`` if empty.
    """
    return _sum([_scaled(c, Var(v.name)) for v, c in expr.items()])


def _decl(v):
    """Convert a PuLP ``LpVariable`` into an IR ``VarDecl``.

    A PuLP binary declares itself as an integer pinned to ``[0, 1]`` — hence the
    ``isBinary`` test precedes ``isInteger``, which is true of both. An absent bound
    is ±Inf; the start is the variable's value if it carries one, else ``0`` clamped
    into the bounds.

    Args:
        v: The PuLP ``LpVariable``.

    Returns:
        VarDecl: The scalar declaration for ``v``.
    """
    domain = BINARY if v.isBinary() else INTEGER if v.isInteger() else CONTINUOUS
    lb = -inf if v.lowBound is None else v.lowBound
    ub = inf if v.upBound is None else v.upBound
    start = v.varValue if v.varValue is not None else min(max(0.0, lb), ub)
    return VarDecl(v.name, [], domain, float(lb), float(ub), float(start))


def import_model(prob):
    """Convert a PuLP ``LpProblem`` into Quicopt's ``Program`` IR.

    Variables are named by their PuLP name, matching ``quicopt.mathopt`` (and unlike
    ``quicopt.pyomo``'s positional names), so solutions line up with what the author
    wrote. A problem with no objective set is a feasibility problem — PuLP's own
    reading — and lowers to a constant ``0`` objective. Two variables sharing a name
    **raise**: PuLP sanitizes names, so distinct variables can collide, and importing
    them would silently merge them into one.
    """
    vars = [_decl(v) for v in prob.variables()]
    names = [d.name for d in vars]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"PuLP variable names are not unique: {dup} — distinct variables "
                         "sharing a name would merge into one; rename them")

    obj = prob.objective
    objective = Const(0.0) if obj is None else _sum([Const(float(obj.constant)), _affine(obj)])
    sense = "max" if prob.sense == pulp.LpMaximize else "min"

    cons = []
    for c in prob.constraints.values():
        lb, ub = _RANGE[c.sense](-float(c.constant))
        _emit(cons, _affine(c), lb, ub)

    return Program(vars=vars, objective=objective, sense=sense, constraints=cons)
