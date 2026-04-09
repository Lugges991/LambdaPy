"""Global name environment for definition resolution during evaluation.

The Haskell evaluator takes a (NameEnv Value_, Env_) pair and resolves
Free_ names from the NameEnv.  Python uses a context-manager-based global
dict so that eval functions don't need an extra parameter.

Usage:
    from lambdapy.nameenv import with_name_env

    with with_name_env(state.definitions):
        result = eval_inf(term, [])
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from lambdapy.syntax import Value

_name_env: dict[str, "Value"] = {}


def get_name_env() -> dict[str, "Value"]:
    """Return the currently active name environment."""
    return _name_env


@contextmanager
def with_name_env(env: dict[str, "Value"]) -> Generator[None, None, None]:
    """Temporarily install *env* as the active name environment."""
    global _name_env
    old = _name_env
    _name_env = env
    try:
        yield
    finally:
        _name_env = old
