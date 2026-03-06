---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Persisted Run Inspection

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

When you pass `run_folder=...` to {meth}`~pipefunc.Pipeline.map` or
{meth}`~pipefunc.Pipeline.map_async`, `pipefunc` persists outputs and metadata
under that directory. This makes it possible to inspect runs after the original
Python process exits, from another terminal, or from another machine that can
see the same filesystem.

`pipefunc-cli` is the built-in command-line tool for that inspection workflow.
It reads run folders on disk and returns JSON, so it works well in shells,
automation, dashboards, and remote debugging sessions.

```{note}
`pipefunc-cli` inspects the folder you point it at. If you copy or restore a
run directory somewhere else, you can inspect the relocated folder directly as
long as its contents are intact.
```

## Inspecting a Single Run

Use `status` to inspect one run folder:

```bash
pipefunc-cli status runs/my-run --pretty
```

This returns a JSON payload with fields such as:

- `status`: one of `pending`, `running`, `incomplete`, `completed`,
  `failed`, or `cancelled`
- `status_source`: `heartbeat` for live async runs or `disk_heuristic` when the
  status is inferred from files on disk
- `progress_fraction`: overall progress when it can be determined
- `outputs`: per-output progress, completion, and byte counts

If you want the raw persisted metadata as well, include `run_info.json` in the
response:

```bash
pipefunc-cli status runs/my-run --include-run-info --pretty
```

## Listing Recent Runs

Use `list-runs` to scan a parent directory such as `runs/`:

```bash
pipefunc-cli list-runs runs --max-runs 20 --pretty
```

This is the fast overview command. It returns one compact summary per run
folder, ordered by recent activity, without including per-output payloads.

That makes it useful for:

- checking whether a shared `runs/` folder contains any active jobs
- seeing which runs completed most recently
- building lightweight monitoring scripts around the JSON output

## Watching a Run

Use `watch` to poll a run folder until it reaches a terminal state:

```bash
pipefunc-cli watch runs/my-run --interval 1 --timeout 300 --pretty
```

`watch` exits with:

- `0` when the run completes successfully
- `1` when the run fails, is cancelled, or cannot be read
- `2` when the timeout is reached

This is useful when `map_async` is running in another process and you want to
tail progress from a second terminal.

## How Status Is Determined

For live async runs, `pipefunc` writes a `pipefunc_status.json` heartbeat. When
that file is present and valid, `pipefunc-cli` prefers it because it can report
live states such as `running`, `failed`, or `cancelled`.

When no heartbeat is available, status is inferred from the persisted run
folder itself:

- scalar outputs are checked by looking for their materialized files
- mapped outputs are inspected via their storage backends
- the overall run is summarized as `pending`, `running`, `incomplete`, or
  `completed`

This fallback is intentionally filesystem-based, so archived, restored, and
completed historical runs remain inspectable even when the original process is
gone.
