# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.pyomo — a Pyomo model → Quicopt ``Program`` IR.

A front-end of the client. Imports at Pyomo's expression-tree level: the
single active objective and each constraint are walked into the IR expression graph,
variable bounds + domain into ``VarDecl``, and function-in-set constraints into
``Zero``/``Nonneg`` rows. The result is a *flat* ``Program`` (one scalar ``VarDecl``
per Pyomo ``VarData``; Pyomo has already expanded indexed components).

An operator absent from the service's catalog is a coverage gap to resolve in the
service, never papered over here.
"""
from __future__ import annotations

from math import inf

import pyomo.environ as pyo

from ._terms import _emit, _sum
from .ir import (Const, Var, Apply, VarDecl, Program,
                 CONTINUOUS, INTEGER, BINARY)

# The ops the service knows. A head outside this set raises — resolve it in the
# service's catalog, do not paper over it here.
_CATALOG = {"+", "-", "*", "/", "^", "sin", "cos", "exp", "log", "sqrt", "abs"}


# ── Pyomo expression node → IR Expression ───────────────────────────────────

def _expr(node, var):
    """Walk a Pyomo expression node into an IR ``Expression``.

    Recurses the Pyomo tree, dispatching on each node's ``getname()`` head: ``sum``
    → n-ary ``+``, ``prod``/``mon`` → ``*``, ``div``/``pow`` → ``/``/``^``, ``neg``
    → ``0 − a`` (the catalog has only binary ``−``), and a unary math head passes
    through under its own catalog name. Variable leaves are mapped by ``var``;
    numeric and parameter leaves are baked to ``Const`` data.

    Args:
        node: A Pyomo expression node, variable, parameter, or Python number.
        var: A callable mapping a Pyomo ``VarData`` to its IR
            :class:`~quicopt.ir.Var`.

    Returns:
        Expression: The corresponding IR expression node.
    """
    if isinstance(node, (int, float)):
        return Const(float(node))
    if node.is_expression_type():
        head = node.getname()
        args = [_expr(a, var) for a in node.args]
        if head == "sum":                       # SumExpression and flattened LinearExpression
            return _sum(args)
        if head in ("prod", "mon"):             # a*b ; and coef*var (both binary)
            return _apply("*", args)
        if head == "div":
            return _apply("/", args)
        if head == "pow":
            return _apply("^", args)
        if head == "neg":                       # unary minus → 0 − a (catalog has binary −)
            return Apply("-", [Const(0.0), args[0]])
        return _apply(head, args)               # a unary math fn reports its catalog name
    if node.is_variable_type():
        return var(node)
    return Const(float(pyo.value(node)))        # constant / parameter leaf → baked to data


def _apply(op, args):
    """Build an ``Apply`` node, guarding the head against the operator catalog.

    Args:
        op: The operator head; must be one of :data:`_CATALOG`.
        args: The IR argument expressions.

    Returns:
        Apply: The ``Apply(op, args)`` node.

    Raises:
        ValueError: If ``op`` is not in the service's operator catalog — a coverage
            gap to resolve in the service, not to paper over here.
    """
    if op not in _CATALOG:
        raise ValueError(f"operator {op!r} is not in the Quicopt operator catalog — "
                         "it must be added to the service before importing")
    return Apply(op, args)


def import_model(m):
    """Convert a Pyomo ``ConcreteModel`` into Quicopt's ``Program`` IR.

    Each ``VarData`` becomes a scalar ``VarDecl`` (``x{i}`` in declaration order)
    carrying its bounds and domain (``Binary``→``BINARY``, integer domains→
    ``INTEGER``, else ``CONTINUOUS``); the active objective and every constraint
    become IR expressions. Requires exactly one active objective; an absent Pyomo
    variable bound is taken as ±Inf (a free direction).
    """
    vis = list(m.component_data_objects(pyo.Var))
    name = {id(v): f"x{i + 1}" for i, v in enumerate(vis)}
    var = lambda v: Var(name[id(v)], ())

    vars = []
    for v in vis:
        domain = BINARY if v.is_binary() else INTEGER if v.is_integer() else CONTINUOUS
        lb, ub = (v.value, v.value) if v.fixed else (v.lb, v.ub)   # a fixed var ⇒ a [val, val] pin
        lb = -inf if lb is None else lb                            # an absent Pyomo bound ⇒ ±Inf (free)
        ub =  inf if ub is None else ub
        start = v.value if v.value is not None else min(max(0.0, lb), ub)   # clamp 0 into [lb, ub]
        vars.append(VarDecl(name[id(v)], [], domain, float(lb), float(ub), float(start)))

    objs = list(m.component_data_objects(pyo.Objective, active=True))
    if len(objs) != 1:
        raise ValueError(f"expected exactly one active objective, found {len(objs)}")
    sense = "min" if objs[0].sense == pyo.minimize else "max"
    objective = _expr(objs[0].expr, var)

    cons = []
    for c in m.component_data_objects(pyo.Constraint, active=True):
        lb = pyo.value(c.lower) if c.has_lb() else -inf   # an absent side ⇒ free, emits no row
        ub = pyo.value(c.upper) if c.has_ub() else inf
        _emit(cons, _expr(c.body, var), lb, ub)

    return Program(vars=vars, objective=objective, sense=sense, constraints=cons)
