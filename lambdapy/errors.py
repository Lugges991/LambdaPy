"""Exception types for LambdaPi."""
from __future__ import annotations


class TypeCheckError(Exception):
    """Raised when type checking fails."""
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class EvalError(Exception):
    """Raised when evaluation encounters an impossible case.
    Should be unreachable for well-typed terms."""
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class ParseError(Exception):
    """Raised when parsing fails."""
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg
