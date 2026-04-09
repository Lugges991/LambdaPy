"""Command-line interface for LambdaPi."""
from __future__ import annotations

import argparse
import sys

from lambdapy.repl import run_file, run_repl


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lambdapy",
        description="LambdaPi: a dependently typed lambda calculus",
    )
    sub = parser.add_subparsers(dest="command")

    # lambdapy repl
    sub.add_parser("repl", help="Start the interactive REPL")

    # lambdapy run FILE
    run_p = sub.add_parser("run", help="Execute a .lp file")
    run_p.add_argument("file", help="Path to .lp file")

    # lambdapy check FILE
    check_p = sub.add_parser("check", help="Type-check a .lp file (same as run)")
    check_p.add_argument("file", help="Path to .lp file")

    args = parser.parse_args()

    match args.command:
        case "run" | "check":
            try:
                run_file(args.file)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
        case "repl" | None:
            run_repl()
        case _:  # pragma: no cover
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
