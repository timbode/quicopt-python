# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.mathopt — an OR-Tools MathOpt model → Quicopt ``Program`` IR.

A second front-end (beside ``quicopt.pyomo``), for authors already holding an
OR-Tools `MathOpt <https://developers.google.com/optimization/math_opt>`_ model.
It walks a ``ModelProto`` — the objective ``offset + Σ cᵢxᵢ + Σ_{i≤j} qᵢⱼxᵢxⱼ`` and
each ``lbᵣ ≤ Σ A[r,j]xⱼ ≤ ubᵣ`` row — into the IR expression graph, with variable
bounds + integrality into ``VarDecl`` and function-in-set rows into
``Zero``/``Nonneg``. ±Inf bounds pass straight through (a free direction); a
one-sided range drops its infinite side.

The supported surface is LP / MILP / (unconstrained binary-quadratic) QUBO — the
classes the service routes. A MathOpt construct outside it (quadratic / second-order
-cone / SOS / indicator constraints, multiple objectives) **raises**: it is a
coverage gap to resolve in the service, never silently dropped here.

Requires the optional ``[mathopt]`` extra (``pip install -e '.[mathopt]'``).
"""
from __future__ import annotations

from ._terms import _emit, _scaled, _sum
from .ir import (Apply, Const, Program, Var, VarDecl,
                 BINARY, CONTINUOUS, INTEGER)

# Constraint kinds a ModelProto can carry that this subset does not serve. Their
# presence raises — the service rejected them too (they are not LP/MILP/QUBO).
_UNSUPPORTED = ("quadratic_constraints", "second_order_cone_constraints",
                "sos1_constraints", "sos2_constraints", "indicator_constraints",
                "auxiliary_objectives")


def import_model(model):
    """Convert an OR-Tools MathOpt model into Quicopt's ``Program`` IR.

    ``model`` is a ``ModelProto`` or a high-level ``mathopt.Model`` (exported to
    its proto). Variables are named by their MathOpt name, falling back to ``x{id}``
    — matching the service's own naming, so solutions line up. A binary variable
    (integer with ``[0,1]`` bounds) maps to ``BINARY``, other integers to
    ``INTEGER``, the rest to ``CONTINUOUS``.
    """
    if hasattr(model, "export_model"):         # a high-level mathopt.Model
        model = model.export_model()

    for fld in _UNSUPPORTED:
        if len(getattr(model, fld)):
            raise ValueError(f"MathOpt {fld} are outside the LP/MILP/QUBO subset the "
                             "service serves — resolve it there, not in the client")

    vp = model.variables
    name = {}                                  # proto var id → IR name
    vars = []
    for k, vid in enumerate(vp.ids):
        nm = vp.names[k] if k < len(vp.names) and vp.names[k] else f"x{vid}"
        name[vid] = nm
        lb, ub = vp.lower_bounds[k], vp.upper_bounds[k]
        integer = k < len(vp.integers) and vp.integers[k]
        domain = BINARY if integer and lb == 0.0 and ub == 1.0 else INTEGER if integer else CONTINUOUS
        start = min(max(0.0, lb), ub)          # 0 clamped into [lb, ub] — finite for any bounds
        vars.append(VarDecl(nm, [], domain, float(lb), float(ub), float(start)))

    o = model.objective
    terms = [Const(float(o.offset))] if o.offset != 0.0 else []
    terms += [_scaled(c, Var(name[vid]))
              for vid, c in zip(o.linear_coefficients.ids, o.linear_coefficients.values)]
    q = o.quadratic_coefficients
    terms += [_scaled(v, Apply("*", [Var(name[r]), Var(name[c])]))   # qᵢⱼ xᵢxⱼ (i≤j, xᵢ² on the diagonal)
              for r, c, v in zip(q.row_ids, q.column_ids, q.coefficients)]
    objective = _sum(terms)
    sense = "max" if o.maximize else "min"

    lcs = model.linear_constraints
    row = {rid: [] for rid in lcs.ids}         # constraint id → its Σ A[r,j]xⱼ terms
    A = model.linear_constraint_matrix
    for r, c, v in zip(A.row_ids, A.column_ids, A.coefficients):
        row[r].append(_scaled(v, Var(name[c])))
    cons = []
    for j, rid in enumerate(lcs.ids):
        _emit(cons, _sum(row[rid]), lcs.lower_bounds[j], lcs.upper_bounds[j])

    return Program(vars=vars, objective=objective, sense=sense, constraints=cons)
