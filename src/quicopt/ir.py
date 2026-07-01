# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.ir — the structured ``Program`` IR.

Front-end-agnostic data: a front-end (``quicopt.pyomo``, ``quicopt.mathopt``) builds a ``Program``;
``quicopt.wire`` serializes it to the versioned wire bytes. The IR/wire is the
service's contract — these types track it, they never fork it. Index tuple entries
are ``int`` (a concrete coordinate) or ``str`` (a bound index symbol); the IR never
uses a bare string for data, so the mapping is total.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

__all__ = [
    "Const", "Param", "Var", "Apply", "Reduce", "SetRef",
    "Zero", "Nonneg", "Indicator",
    "VarDecl", "IndexSet", "Constraint", "Program",
    "Domain", "CONTINUOUS", "INTEGER", "BINARY",
]


class Domain(IntEnum):
    """The domain a variable ranges over. Values are the wire enum codes."""
    CONTINUOUS = 1
    INTEGER = 2
    BINARY = 3


CONTINUOUS = Domain.CONTINUOUS
INTEGER = Domain.INTEGER
BINARY = Domain.BINARY


# ── Expression = Const | Param | Var | Apply | Reduce ───────────────────────

class Expression:
    """Base of the expression grammar."""


@dataclass
class Const(Expression):
    """A literal numeric constant."""
    value: float


@dataclass
class Param(Expression):
    """A reference to a parameter-table entry — data bound at instance time, by
    ``name`` and an index tuple (``()`` for a scalar)."""
    name: str
    index: tuple = ()


@dataclass
class Var(Expression):
    """A reference to a decision variable, by ``name`` and an index tuple
    (``()`` for a scalar)."""
    name: str
    index: tuple = ()


@dataclass
class Apply(Expression):
    """A catalog operator ``op`` applied to its argument subexpressions."""
    op: str
    args: list                  # list[Expression]


@dataclass
class SetRef:
    """A reference to an index set: ``args=()`` is the flat set; ``args=("i",)``
    references the set indexed by enclosing bound indices."""
    name: str
    args: tuple = ()


@dataclass
class Reduce(Expression):
    """A fold of ``body`` over ``idx`` ranging across ``over`` — e.g. a Σ or Π —
    keeping a term only where ``cond`` is non-zero (``None`` ⇒ keep every term)."""
    op: str                     # the fold operator key (+, *, …)
    idx: str                    # the bound dummy index
    over: SetRef
    body: Expression
    cond: "Expression | None" = None    # keep a term only where cond ≠ 0


# ── ConSet = Zero | Nonneg | Indicator ──────────────────────────────────────

@dataclass
class Zero:
    """f = 0"""


@dataclass
class Nonneg:
    """f ≥ 0"""


@dataclass
class Indicator:
    """``bin`` active (= 1) implies the body satisfies the ``inner`` ConSet."""
    bin: Var                    # the binary being active implies f ∈ inner
    inner: object               # ConSet


# ── variables, constraints, container ───────────────────────────────────────

@dataclass
class VarDecl:
    """A variable declaration: ``name`` over the product of the index sets in
    ``axes`` (``[]`` ⇒ scalar), ranging over ``domain``, with ``lower``/``upper``
    bounds and an initial ``start``. A bound is a float, or the ``str`` name of a
    Param table when it varies by index."""
    name: str
    axes: list                  # list[str] of set names; [] ⇒ scalar
    domain: Domain
    lower: object               # float scalar, or str = name of a Param table
    upper: object
    start: float


@dataclass
class IndexSet:
    """A named index set with concrete ``elements`` (each ``int`` or ``str``)."""
    name: str
    elements: list              # list[int | str]


@dataclass
class Constraint:
    """A constraint row: the expression ``f`` lies in the ConSet ``set``, for every
    binding in ``over`` (``[(dummy, SetRef)]``; empty ⇒ a single scalar row)."""
    f: Expression
    set: object                 # ConSet
    over: list = field(default_factory=list)    # list[(str, SetRef)] quantifier bindings


@dataclass
class Program:
    """A complete optimization model: index sets and data tables, variable
    declarations, the objective and its ``sense``, the constraint rows, and any
    per-index variable pins (``fix``)."""
    sets: list = field(default_factory=list)          # list[IndexSet]
    indexed_sets: dict = field(default_factory=dict)  # {name: {tuple: list}}  a[(i,)] -> [j…]
    params: dict = field(default_factory=dict)        # {name: {tuple: float}} p[(i,)], A[(i,j)]
    vars: list = field(default_factory=list)          # list[VarDecl]
    objective: Expression = None
    sense: str = "min"                                # "min" | "max"
    constraints: list = field(default_factory=list)   # list[Constraint]
    fix: dict = field(default_factory=dict)           # {(varname, tuple): float} per-index pins
