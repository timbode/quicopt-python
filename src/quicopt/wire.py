# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.wire — a structured ``Program`` → versioned wire bytes.

A hand-rolled, stdlib-only encoder of Quicopt's wire schema (the deliberate "no
protoc" choice: the schema is small and frozen). Crucially it is **byte-identical**
to what the Quicopt service expects: the same canonical ordering of the order-free
``dict`` tables (so equal Programs encode to equal bytes), the same always-emit-the-
oneof-case rule (so ``Const(0.0)`` survives), and the same always-emit scalars
(``VarDecl.start``, ``ParamEntry.value`` even when ``0.0`` — which proto3's
default-omission would drop). That exactness is the encoder's correctness property
(see ``tests/test_wire_golden.py``).

Wire format crib (proto3): a field is ``tag = (number<<3 | wiretype)`` then the
payload; wiretype 0=varint, 1=fixed64 (doubles, little-endian), 2=length-delimited
(strings, embedded messages, repeated). Embedded messages are length-delimited.
"""
from __future__ import annotations

import struct
from io import BytesIO

from .ir import (Const, Param, Var, Apply, Reduce, SetRef,
                 Zero, Nonneg, Indicator, VarDecl, IndexSet, Program)

SCHEMA_VERSION = 1               # co-versions with the .proto package (…v1)


# ── low-level writers ───────────────────────────────────────────────────────

def _putvarint(io, x):
    """Write an integer as a base-128 varint (LEB128, proto wiretype 0).

    Args:
        io: A byte sink (``BytesIO``) the encoded bytes are written to.
        x: The integer to encode. Negatives are taken as their two's-complement
            in 64 bits, matching proto's ``int64`` (so ``-1`` becomes a 10-byte
            varint), which is why the domain is not just non-negative.

    Returns:
        None. The bytes are appended to ``io`` in place.
    """
    x = int(x)
    if x < 0:                    # two's-complement into 64 bits (−1 ⇒ a 10-byte varint)
        x &= (1 << 64) - 1
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            io.write(bytes((b | 0x80,)))
        else:
            io.write(bytes((b,)))
            return


def _puttag(io, field, wt):
    """Write a field's key tag: the varint ``(field << 3) | wiretype``.

    Args:
        io: The byte sink.
        field: The proto field number.
        wt: The wiretype (0=varint, 1=fixed64, 2=length-delimited).

    Returns:
        None. The tag is appended to ``io``.
    """
    _putvarint(io, (field << 3) | wt)


# ── field writers ───────────────────────────────────────────────────────────

def _wbytes(io, field, b):
    """Write a length-delimited (wiretype 2) field: tag, then length, then bytes.

    Args:
        io: The byte sink.
        field: The proto field number.
        b: The raw payload bytes (a string, an embedded message, or packed data).

    Returns:
        None. The field is appended to ``io``.
    """
    _puttag(io, field, 2)
    _putvarint(io, len(b))
    io.write(b)


def _wstr(io, field, s):
    """Write a string field as its UTF-8 bytes (a length-delimited field).

    Args:
        io: The byte sink.
        field: The proto field number.
        s: The value to encode; coerced via ``str`` so a ``Symbol``-like object
            is accepted, then UTF-8 encoded.

    Returns:
        None. The field is appended to ``io``.
    """
    _wbytes(io, field, str(s).encode("utf-8"))      # Symbol/str → UTF-8


def _wmsg(io, field, b):
    """Write an embedded-message field (a length-delimited field over its bytes).

    Semantically distinct from :func:`_wbytes` — it names the intent (an embedded
    message rather than opaque bytes) though the encoding is identical.

    Args:
        io: The byte sink.
        field: The proto field number.
        b: The already-serialized bytes of the embedded message.

    Returns:
        None. The field is appended to ``io``.
    """
    _wbytes(io, field, b)


def _wdouble(io, field, x):
    """Write a ``double`` field as little-endian fixed64 (wiretype 1).

    ±Inf and NaN round-trip as their IEEE-754 bit patterns, which is how
    unbounded variable bounds travel over the wire.

    Args:
        io: The byte sink.
        field: The proto field number.
        x: The value to encode; coerced via ``float``.

    Returns:
        None. The 8-byte field is appended to ``io``.
    """
    _puttag(io, field, 1)
    io.write(struct.pack("<d", float(x)))


def _wvarint(io, field, x):
    """Write an integer field as a varint (wiretype 0).

    Args:
        io: The byte sink.
        field: The proto field number.
        x: The integer to encode (see :func:`_putvarint` for negatives).

    Returns:
        None. The field is appended to ``io``.
    """
    _puttag(io, field, 0)
    _putvarint(io, x)


def _msg(build):
    """Materialize an embedded message's bytes by running its builder on a buffer.

    The one-line idiom for "serialize a sub-message": hand a fresh sink to
    ``build`` and hand back what it wrote, ready to pass to :func:`_wmsg`.

    Args:
        build: A callable ``build(io)`` that writes the message's fields into the
            sink it is given.

    Returns:
        bytes: The serialized message.
    """
    io = BytesIO()
    build(io)
    return io.getvalue()


# ── canonical ordering (deterministic bytes from order-free dicts) ──────────
#   Vectors (sets, vars, constraints, args) keep their order; dict-keyed tables
#   (params, indexed_sets, fix) are sorted so two encodes of equal data agree.
#   the _elemkey/_idxkey comparator (int < symbol at each position).

def _elemkey(e):
    """Sort key for a single index element, ordering all ints before all symbols.

    An index coordinate is either an ``int`` (a concrete position) or a ``str`` (a
    bound index symbol). Returning ``(0, float(e), "")`` for ints and
    ``(1, 0.0, str(e))`` for strings puts every int ahead of every symbol and
    breaks ties within each kind numerically / lexicographically — the ordering
    that makes the encoded bytes canonical.

    Args:
        e: An index element, ``int`` or ``str`` (``bool`` is deliberately *not*
            treated as ``int`` here).

    Returns:
        tuple: A ``(kind, number, text)`` sort key.
    """
    if isinstance(e, int) and not isinstance(e, bool):
        return (0, float(e), "")
    return (1, 0.0, str(e))


def _idxkey(t):
    """Sort key for an index *tuple*, ordered by length then element-wise.

    Args:
        t: An index tuple (a key of ``indexed_sets`` / ``fix`` fibres).

    Returns:
        tuple: ``(len(t), (elemkey, …))``, so shorter tuples sort first and equal
        lengths compare position by position under :func:`_elemkey`.
    """
    return (len(t), tuple(_elemkey(e) for e in t))


# ── message encoders ────────────────────────────────────────────────────────

def _enc_index_elem(io, e):
    """Encode one index element into an ``IndexElem`` message body.

    The element's Python type selects the ``oneof`` case: a symbol goes to field 2
    (string), a concrete coordinate to field 1 (varint).

    Args:
        io: The byte sink.
        e: The index element, ``str`` (a symbol) or ``int`` (a coordinate).

    Returns:
        None. The fields are appended to ``io``.

    Raises:
        TypeError: If ``e`` is neither ``int`` nor ``str`` (``bool`` is rejected).
    """
    if isinstance(e, str):
        _wstr(io, 2, e)
    elif isinstance(e, int) and not isinstance(e, bool):
        _wvarint(io, 1, e)
    else:
        raise TypeError(f"wire: index element must be int or str, got {type(e).__name__}: {e!r}")


def _enc_index(io, idx):
    """Encode an index tuple as a repeated field of ``IndexElem`` messages.

    Args:
        io: The byte sink.
        idx: The index tuple whose elements are emitted in order under field 1.

    Returns:
        None. The repeated fields are appended to ``io``.
    """
    for e in idx:
        _wmsg(io, 1, _msg(lambda b, e=e: _enc_index_elem(b, e)))


def _idx_msg(idx):
    """Serialize an index tuple to its standalone ``Index`` message bytes.

    Args:
        idx: The index tuple.

    Returns:
        bytes: The encoded ``Index`` message, for embedding via :func:`_wmsg`.
    """
    return _msg(lambda b: _enc_index(b, idx))


def _enc_var_ref(io, v):
    """Encode a ``Var`` reference (name + index) into a ``VarRef`` message body.

    Args:
        io: The byte sink.
        v: The :class:`~quicopt.ir.Var` to encode (``v.name``, ``v.index``).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, v.name)
    _wmsg(io, 2, _idx_msg(v.index))


def _enc_param_ref(io, p):
    """Encode a ``Param`` reference (name + index) into a ``ParamRef`` message body.

    Args:
        io: The byte sink.
        p: The :class:`~quicopt.ir.Param` to encode (``p.name``, ``p.index``).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, p.name)
    _wmsg(io, 2, _idx_msg(p.index))


def _enc_set_ref(io, s):
    """Encode a ``SetRef`` (name + args) into a ``SetRef`` message body.

    Args:
        io: The byte sink.
        s: The :class:`~quicopt.ir.SetRef` to encode (``s.name``, ``s.args``).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, s.name)
    _wmsg(io, 2, _idx_msg(s.args))


def _enc_expr(io, e):
    """Encode an expression node into an ``Expression`` message body.

    The node's concrete type selects the active ``oneof`` case — ``Const``→1
    (double), ``Param``→2, ``Var``→3, ``Apply``→4 (op + args), ``Reduce``→5
    (op, bound index, set, body, optional guard). The case is *always* emitted so
    that ``Const(0.0)`` survives round-trip rather than collapsing to an empty
    message.

    Args:
        io: The byte sink.
        e: An IR expression node (:class:`~quicopt.ir.Const` /
            :class:`~quicopt.ir.Param` / :class:`~quicopt.ir.Var` /
            :class:`~quicopt.ir.Apply` / :class:`~quicopt.ir.Reduce`).

    Returns:
        None. The fields are appended to ``io``.

    Raises:
        TypeError: If ``e`` is not one of the expression node types.
    """
    if isinstance(e, Const):
        _wdouble(io, 1, e.value)
    elif isinstance(e, Param):
        _wmsg(io, 2, _msg(lambda b: _enc_param_ref(b, e)))
    elif isinstance(e, Var):
        _wmsg(io, 3, _msg(lambda b: _enc_var_ref(b, e)))
    elif isinstance(e, Apply):
        def build(b):
            _wstr(b, 1, e.op)
            for a in e.args:
                _wmsg(b, 2, _expr_msg(a))
        _wmsg(io, 4, _msg(build))
    elif isinstance(e, Reduce):
        def build(b):
            _wstr(b, 1, e.op)
            _wstr(b, 2, e.idx)
            _wmsg(b, 3, _msg(lambda c: _enc_set_ref(c, e.over)))
            _wmsg(b, 4, _expr_msg(e.body))
            if e.cond is not None:
                _wmsg(b, 5, _expr_msg(e.cond))
        _wmsg(io, 5, _msg(build))
    else:
        raise TypeError(f"wire: not an Expression: {e!r}")


def _expr_msg(e):
    """Serialize an expression node to its standalone ``Expression`` message bytes.

    Args:
        e: An IR expression node.

    Returns:
        bytes: The encoded ``Expression`` message, for embedding via :func:`_wmsg`.
    """
    return _msg(lambda b: _enc_expr(b, e))


def _enc_conset(io, c):
    """Encode a constraint set into a ``ConSet`` message body.

    The ``oneof`` case marks the set the constraint body ``f`` must lie in:
    ``Zero``→1 (``f == 0``) and ``Nonneg``→2 (``f >= 0``) are empty markers;
    ``Indicator``→3 nests a binary ``VarRef`` and an inner ``ConSet`` (the
    constraint that is active when the binary is on).

    Args:
        io: The byte sink.
        c: A constraint set (:class:`~quicopt.ir.Zero` /
            :class:`~quicopt.ir.Nonneg` / :class:`~quicopt.ir.Indicator`).

    Returns:
        None. The fields are appended to ``io``.

    Raises:
        TypeError: If ``c`` is not one of the constraint-set types.
    """
    if isinstance(c, Zero):
        _wmsg(io, 1, b"")
    elif isinstance(c, Nonneg):
        _wmsg(io, 2, b"")
    elif isinstance(c, Indicator):
        def build(b):
            _wmsg(b, 1, _msg(lambda c2: _enc_var_ref(c2, c.bin)))
            _wmsg(b, 2, _msg(lambda c2: _enc_conset(c2, c.inner)))
        _wmsg(io, 3, _msg(build))
    else:
        raise TypeError(f"wire: not a ConSet: {c!r}")


def _conset_msg(s):
    """Serialize a constraint set to its standalone ``ConSet`` message bytes.

    Args:
        s: A constraint set (``Zero`` / ``Nonneg`` / ``Indicator``).

    Returns:
        bytes: The encoded ``ConSet`` message, for embedding via :func:`_wmsg`.
    """
    return _msg(lambda b: _enc_conset(b, s))


def _enc_constraint(io, c):
    """Encode a ``Constraint`` (body, set, quantifiers) into its message body.

    Field 1 is the body expression ``f``, field 2 the constraint set it must lie
    in, and field 3 a repeated ``(symbol, SetRef)`` quantifier — the ``∀`` binders
    that range the constraint over an index set.

    Args:
        io: The byte sink.
        c: The :class:`~quicopt.ir.Constraint` to encode (``c.f``, ``c.set``,
            ``c.over``).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wmsg(io, 1, _expr_msg(c.f))
    _wmsg(io, 2, _conset_msg(c.set))
    for (sym, sref) in c.over:
        def build(b, sym=sym, sref=sref):
            _wstr(b, 1, sym)
            _wmsg(b, 2, _msg(lambda c2: _enc_set_ref(c2, sref)))
        _wmsg(io, 3, _msg(build))


def _enc_bound(io, b):
    """Encode a variable bound into a ``Bound`` message body (``oneof`` numeric/ref).

    A bound is either a literal number (field 1, a ``double`` — ±Inf for an open
    direction) or the name of a ``Param`` table that supplies it (field 2).

    Args:
        io: The byte sink.
        b: The bound: a ``str`` (a param-table name) or a number.

    Returns:
        None. The field is appended to ``io``.
    """
    if isinstance(b, str):       # a Param-table name
        _wstr(io, 2, b)
    else:
        _wdouble(io, 1, b)


def _bound_msg(b):
    """Serialize a variable bound to its standalone ``Bound`` message bytes.

    Args:
        b: The bound (a param-table name ``str`` or a number).

    Returns:
        bytes: The encoded ``Bound`` message, for embedding via :func:`_wmsg`.
    """
    return _msg(lambda c: _enc_bound(c, b))


def _enc_var_decl(io, vd):
    """Encode a ``VarDecl`` into its message body.

    Emits name (1), the repeated axis names (2), the integer domain code (3), the
    lower/upper ``Bound`` messages (4, 5), and the ``start`` value (6). The start
    is *always* emitted even when ``0.0`` so it survives proto3 default-omission.

    Args:
        io: The byte sink.
        vd: The :class:`~quicopt.ir.VarDecl` to encode (``name``, ``axes``,
            ``domain``, ``lower``, ``upper``, ``start``).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, vd.name)
    for ax in vd.axes:
        _wstr(io, 2, ax)
    _wvarint(io, 3, int(vd.domain))
    _wmsg(io, 4, _bound_msg(vd.lower))
    _wmsg(io, 5, _bound_msg(vd.upper))
    _wdouble(io, 6, vd.start)


def _enc_index_set(io, s):
    """Encode an ``IndexSet`` (name + explicit element list) into its message body.

    Args:
        io: The byte sink.
        s: The :class:`~quicopt.ir.IndexSet` to encode (``s.name``, ``s.elements``);
            elements are emitted in their given order (field 2, repeated).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, s.name)
    for el in s.elements:
        _wmsg(io, 2, _msg(lambda b, el=el: _enc_index_elem(b, el)))


def _enc_indexed_set(io, name, fibres):
    """Encode one indexed (parametrized) set: name + its key→elements fibres.

    Each fibre is an entry ``(key index, element list)``. The fibres are emitted
    in :func:`_idxkey` order so this order-free ``dict`` produces canonical bytes.

    Args:
        io: The byte sink.
        name: The set's name.
        fibres: A mapping ``index tuple → iterable of index elements`` (the members
            of the set at that key).

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, name)
    for key in sorted(fibres.keys(), key=_idxkey):
        def build(b, key=key):
            _wmsg(b, 1, _idx_msg(key))
            for el in fibres[key]:
                _wmsg(b, 2, _msg(lambda c, el=el: _enc_index_elem(c, el)))
        _wmsg(io, 2, _msg(build))


def _enc_param_table(io, name, tbl):
    """Encode one ``Param`` table: name + its key→value entries.

    Entries are emitted in :func:`_idxkey` order (canonical bytes from an order-free
    ``dict``), and each value is *always* written even when ``0.0``.

    Args:
        io: The byte sink.
        name: The parameter table's name.
        tbl: A mapping ``index tuple → double`` of parameter values.

    Returns:
        None. The fields are appended to ``io``.
    """
    _wstr(io, 1, name)
    for key in sorted(tbl.keys(), key=_idxkey):
        def build(b, key=key):
            _wmsg(b, 1, _idx_msg(key))
            _wdouble(b, 2, tbl[key])
        _wmsg(io, 2, _msg(build))


def encode(prog: Program) -> bytes:
    """Serialize a ``Program`` to versioned (v1) wire bytes.

    The top-level message in schema order: index sets (1), indexed sets (2),
    param tables (3), variable declarations (4), the objective expression (5), the
    sense string (6), constraints (7), and the ``fix`` pins (8). The order-free
    tables (``indexed_sets``, ``params``, ``fix``) are emitted in sorted-key order
    so equal Programs encode to equal bytes — the byte-identity property the golden
    test pins.

    Args:
        prog: The :class:`~quicopt.ir.Program` to serialize.

    Returns:
        bytes: The wire message, deterministic and byte-identical to what the
        Quicopt service decodes for the same model.
    """
    io = BytesIO()
    for s in prog.sets:
        _wmsg(io, 1, _msg(lambda b, s=s: _enc_index_set(b, s)))
    for name in sorted(prog.indexed_sets.keys()):
        _wmsg(io, 2, _msg(lambda b, name=name: _enc_indexed_set(b, name, prog.indexed_sets[name])))
    for name in sorted(prog.params.keys()):
        _wmsg(io, 3, _msg(lambda b, name=name: _enc_param_table(b, name, prog.params[name])))
    for vd in prog.vars:
        _wmsg(io, 4, _msg(lambda b, vd=vd: _enc_var_decl(b, vd)))
    _wmsg(io, 5, _expr_msg(prog.objective))
    _wstr(io, 6, prog.sense)
    for c in prog.constraints:
        _wmsg(io, 7, _msg(lambda b, c=c: _enc_constraint(b, c)))
    for key in sorted(prog.fix.keys(), key=lambda k: (k[0], _idxkey(k[1]))):
        var, idx = key
        def build(b, var=var, idx=idx, key=key):
            _wstr(b, 1, var)
            _wmsg(b, 2, _idx_msg(idx))
            _wdouble(b, 3, prog.fix[key])
        _wmsg(io, 8, _msg(build))
    return io.getvalue()


def encode_params(params: dict) -> bytes:
    """Serialize just the ``Param`` tables as a standalone ``ParamData`` message.

    This supports rebinding parameters over the wire without resending the model:
    graph the ``Program`` once, then emit one ``ParamData`` per instance. Tables are
    emitted in sorted-key order to stay byte-canonical.

    Args:
        params: A mapping ``table name → (index tuple → double)`` of the parameter
            values to send.

    Returns:
        bytes: The encoded ``ParamData`` message.
    """
    io = BytesIO()
    for name in sorted(params.keys()):
        _wmsg(io, 1, _msg(lambda b, name=name: _enc_param_table(b, name, params[name])))
    return io.getvalue()
