# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt — the Python client for the Quicopt optimization service.

Authors an optimization model in a Python front-end (Pyomo, …) and converts it to
Quicopt's wire IR — the versioned, language-neutral contract the service consumes.
This package is a thin front-end; see ``README.md`` for usage.

Layers:
  ir      — the ``Program`` IR data model (front-end-agnostic)
  wire    — ``Program`` → versioned wire bytes (front-end-agnostic)
  pyomo   — a Pyomo model → ``Program`` importer (a front-end)
  mathopt — an OR-Tools MathOpt model → ``Program`` importer (a front-end)
  client  — POST the wire bytes to the service and read the result (HTTP, stdlib)
"""
from .ir import (Const, Param, Var, Apply, Reduce, SetRef,
                 Zero, Nonneg, Indicator,
                 VarDecl, IndexSet, Constraint, Program,
                 Domain, CONTINUOUS, INTEGER, BINARY)
from .wire import encode, encode_params, SCHEMA_VERSION
from .client import Client, Job, Result, QuicoptError

__all__ = [
    "Const", "Param", "Var", "Apply", "Reduce", "SetRef",
    "Zero", "Nonneg", "Indicator",
    "VarDecl", "IndexSet", "Constraint", "Program",
    "Domain", "CONTINUOUS", "INTEGER", "BINARY",
    "encode", "encode_params", "SCHEMA_VERSION",
    "Client", "Job", "Result", "QuicoptError",
]
