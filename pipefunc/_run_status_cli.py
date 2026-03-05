"""CLI for inspecting persisted pipefunc runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING, Any

from pipefunc._run_status import list_run_statuses, status_from_run_folder

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the status CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        payload = status_from_run_folder(
            args.run_folder,
            include_outputs=True,
            include_run_info=args.include_run_info,
        )
        _print_json(payload, pretty=args.pretty)
        return 1 if "error" in payload else 0
    if args.command == "list-runs":
        payload = list_run_statuses(folder=args.folder, max_runs=args.max_runs)
        _print_json(payload, pretty=args.pretty)
        return 1 if payload["error"] is not None else 0
    if args.command == "watch":
        return _watch_command(args)
    parser.error(f"Unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect persisted pipefunc run folders as JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser(
        "status",
        help="Show the current status of a run folder.",
    )
    status_parser.add_argument("run_folder", help="Path to a pipefunc run folder.")
    status_parser.add_argument(
        "--include-run-info",
        action="store_true",
        help="Include raw run_info.json content in the JSON payload.",
    )
    status_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )

    list_parser = subparsers.add_parser(
        "list-runs",
        help="List run folders under a parent directory.",
    )
    list_parser.add_argument(
        "folder",
        nargs="?",
        default="runs",
        help="Parent directory containing run folders. Defaults to 'runs'.",
    )
    list_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of runs returned.",
    )
    list_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Poll a run folder until it completes or times out.",
    )
    watch_parser.add_argument("run_folder", help="Path to a pipefunc run folder.")
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds. Defaults to 5.",
    )
    watch_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds. Exits with code 2 on timeout.",
    )
    return parser


def _watch_command(args: argparse.Namespace) -> int:
    started = time.monotonic()
    while True:
        payload = status_from_run_folder(args.run_folder, include_outputs=True)
        _print_json(payload, pretty=False)
        if "error" in payload:
            return 1
        if payload["status"] == "completed":
            return 0
        if payload["status"] in {"cancelled", "failed"}:
            return 1
        if args.timeout is not None and (time.monotonic() - started) >= args.timeout:
            return 2
        time.sleep(args.interval)


def _print_json(payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    else:
        json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
