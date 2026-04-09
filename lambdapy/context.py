"""Context for type checking.

A Context maps Names to either a type (HasType) or a kind marker (HasKind).
It is represented as an association list (newest binding first) so that
prepending is O(1) and shadowing works correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from lambdapy.errors import TypeCheckError
from lambdapy.syntax import Name, Value


@dataclass(frozen=True)
class HasType:
    """The name has the given value-level type."""
    type: Value


@dataclass(frozen=True)
class HasKind:
    """The name is a type variable (used in STLC-style contexts)."""


ContextEntry = HasType | HasKind

# A context is an association list: list of (Name, ContextEntry) pairs.
# The most recently added binding comes first.
Context = list[tuple[Name, ContextEntry]]


def context_lookup(ctx: Context, name: Name) -> ContextEntry:
    """Look up a name in the context.  Raises TypeCheckError if not found."""
    for n, entry in ctx:
        if n == name:
            return entry
    raise TypeCheckError(f"Unknown identifier: {name!r}")


def empty_context() -> Context:
    """Return a fresh empty context."""
    return []
