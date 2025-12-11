Archive directory containing relocated/archived files

Purpose:
- Keep recoverable copies of notebooks and helper scripts moved out of main workflow.
- Original files replaced by lightweight stubs in `scripts/` that forward to `src/` or point to archived copies.

How to restore a file:
1. Copy the file from `archive/scripts/<name>.py` back into `scripts/`.
2. Remove the stub from `scripts/` if present.

Notes:
- Core analysis code has been moved to `src/analysis/` and the dashboard to `src/dashboard/`.
- Use the top-level entry points (same filenames) — the stubs will forward execution to `src/`.
