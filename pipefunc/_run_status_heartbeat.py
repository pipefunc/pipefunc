"""Persist and load live run-status heartbeats for async pipeline runs."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status
    from pipefunc.map._run_info import RunInfo

RUN_STATUS_FILENAME = "pipefunc_status.json"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
HEARTBEAT_STALE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class FunctionStatusBinding:
    """Describe which function produces which output names."""

    function_name: str
    output_names: tuple[str, ...]


@dataclass
class RunStatusHeartbeatWriter:
    """Write periodic run-status heartbeats for a persisted async run."""

    run_info: RunInfo
    progress_dict: Mapping[OUTPUT_TYPE, Status]
    function_bindings: Mapping[OUTPUT_TYPE, FunctionStatusBinding]
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    _pipeline_task: asyncio.Task[Any] | None = None
    _heartbeat_task: asyncio.Task[None] | None = None
    _event_loop: asyncio.AbstractEventLoop | None = None
    _write_scheduled: bool = False
    _schedule_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        """Start heartbeat writes for the provided pipeline task."""
        self._pipeline_task = task
        self._event_loop = task.get_loop()
        for status in self.progress_dict.values():
            status.add_update_callback(self.request_write)
        self.write_now(task=task)
        task.add_done_callback(self._on_task_done)
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def write_now(self, *, task: asyncio.Task[Any] | None = None) -> None:
        """Write the current heartbeat snapshot immediately."""
        run_folder = self.run_info.run_folder
        assert run_folder is not None
        payload = build_run_status_snapshot(
            run_info=self.run_info,
            progress_dict=self.progress_dict,
            function_bindings=self.function_bindings,
            task=task or self._pipeline_task,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
        )
        _write_json_atomic(run_status_path(run_folder), payload)

    async def _heartbeat_loop(self) -> None:
        try:
            while self._pipeline_task is not None and not self._pipeline_task.done():
                await asyncio.sleep(self.heartbeat_interval_seconds)
                self.write_now()
        except asyncio.CancelledError:
            pass

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        self.write_now(task=task)

    def request_write(self) -> None:
        """Coalesce status-triggered writes onto the pipeline event loop."""
        loop = self._event_loop
        if loop is None:
            return
        with self._schedule_lock:
            if self._write_scheduled:
                return
            self._write_scheduled = True
        try:
            loop.call_soon_threadsafe(self._flush_scheduled_write)
        except RuntimeError:
            with self._schedule_lock:
                self._write_scheduled = False

    def _flush_scheduled_write(self) -> None:
        with self._schedule_lock:
            self._write_scheduled = False
        self.write_now()


def run_status_path(run_folder: str | Path) -> Path:
    """Return the persisted heartbeat path for a run folder."""
    return Path(run_folder) / RUN_STATUS_FILENAME


def load_run_status_heartbeat(run_folder: str | Path) -> dict[str, Any] | None:
    """Load a persisted heartbeat payload if present."""
    path = run_status_path(run_folder)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def add_heartbeat_staleness(
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Annotate a heartbeat payload with staleness information."""
    if payload.get("status") in {"cancelled", "completed", "failed"}:
        payload["stale"] = False
        return payload
    updated_at_raw = payload.get("updated_at")
    updated_at = _parse_iso_timestamp(updated_at_raw) if isinstance(updated_at_raw, str) else None
    stale_after_seconds = payload.get("stale_after_seconds")
    if updated_at is None or not isinstance(stale_after_seconds, int | float):
        payload["stale"] = False
        return payload
    current_time = now or datetime.now(tz=timezone.utc)
    payload["stale"] = (current_time - updated_at).total_seconds() > float(stale_after_seconds)
    return payload


def build_run_status_snapshot(
    *,
    run_info: RunInfo,
    progress_dict: Mapping[OUTPUT_TYPE, Status],
    function_bindings: Mapping[OUTPUT_TYPE, FunctionStatusBinding],
    task: asyncio.Task[Any] | None,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Build a JSON-serializable snapshot for a live run."""
    updated_at = _isoformat_now()
    outputs: dict[str, Any] = {}
    functions: list[dict[str, Any]] = []
    active_functions: list[str] = []
    active_outputs: list[str] = []
    seen_active_functions: set[str] = set()

    for output_key, status in progress_dict.items():
        binding = function_bindings.get(output_key) or FunctionStatusBinding(
            function_name="-".join(at_least_tuple(output_key)),
            output_names=at_least_tuple(output_key),
        )
        function_payload = _function_payload(binding, status)
        functions.append(function_payload)
        if function_payload["state"] == "running":
            active_outputs.extend(binding.output_names)
            if binding.function_name not in seen_active_functions:
                active_functions.append(binding.function_name)
                seen_active_functions.add(binding.function_name)
        output_payload = _output_payload(binding.function_name, status)
        for output_name in binding.output_names:
            outputs[output_name] = output_payload

    all_complete = all(output["complete"] for output in outputs.values())
    progress_fraction = _overall_progress(outputs)
    return {
        "run_folder": str(run_info.run_folder.absolute()) if run_info.run_folder else None,
        "status": _derive_run_status(
            task=task,
            all_complete=all_complete,
            outputs=outputs,
        ),
        "status_source": "heartbeat",
        "all_complete": all_complete,
        "progress_fraction": progress_fraction,
        "n_outputs": len(outputs),
        "n_outputs_completed": sum(1 for output in outputs.values() if output["complete"]),
        "outputs": outputs,
        "functions": functions,
        "active_outputs": active_outputs,
        "active_functions": active_functions,
        "updated_at": updated_at,
        "last_modified": updated_at,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "stale_after_seconds": heartbeat_interval_seconds * HEARTBEAT_STALE_MULTIPLIER,
        "pipefunc_version": run_info.pipefunc_version,
    }


def _function_payload(binding: FunctionStatusBinding, status: Status) -> dict[str, Any]:
    elapsed_time = status.elapsed_time()
    return {
        "function_name": binding.function_name,
        "output_names": list(binding.output_names),
        "state": _status_state(status),
        "progress": status.progress,
        "complete": status.progress >= 1.0,
        "n_total": status.n_total,
        "n_in_progress": status.n_in_progress,
        "n_completed": status.n_completed,
        "n_failed": status.n_failed,
        "elapsed_time": elapsed_time,
        "remaining_time": status.remaining_time(elapsed_time=elapsed_time),
    }


def _output_payload(function_name: str, status: Status) -> dict[str, Any]:
    elapsed_time = status.elapsed_time()
    return {
        "function_name": function_name,
        "progress": status.progress,
        "complete": status.progress >= 1.0,
        "n_total": status.n_total,
        "n_in_progress": status.n_in_progress,
        "n_completed": status.n_completed,
        "n_failed": status.n_failed,
        "elapsed_time": elapsed_time,
        "remaining_time": status.remaining_time(elapsed_time=elapsed_time),
    }


def _status_state(status: Status) -> str:
    if status.progress >= 1.0:
        return "completed"
    if status.n_in_progress > 0 or status.start_time is not None or status.n_attempted > 0:
        return "running"
    return "pending"


def _derive_run_status(
    *,
    task: asyncio.Task[Any] | None,
    all_complete: bool,
    outputs: dict[str, Any],
) -> str:
    if task is not None:
        if task.cancelled():
            return "cancelled"
        if task.done():
            return "failed" if task.exception() is not None else "completed"
    if all_complete:
        return "completed"
    if any(output["n_in_progress"] > 0 for output in outputs.values()):
        return "running"
    if any(output["progress"] > 0 for output in outputs.values()):
        return "running"
    return "pending"


def _overall_progress(outputs: dict[str, Any]) -> float | None:
    values = [
        float(output["progress"])
        for output in outputs.values()
        if isinstance(output["progress"], int | float)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{time_ns()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def _isoformat_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_iso_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def time_ns() -> int:
    """Small wrapper to simplify deterministic monkeypatching in tests."""
    return time.time_ns()
