# KRSL AI

Local-first research and MVP project for recognition of isolated
Kazakh-Russian Sign Language (KRSL) signs. The first milestone is a
reproducible signer-independent KRSL20 recognizer, not continuous sign-language
translation.

## Current status

Phase 1 is in progress: KRSL20 inventory and signer-independent split design.
The project foundation and legacy audit are complete; no model or dataset
preprocessing code is included yet.

## Local setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Raw videos and extracted keypoints are intentionally excluded from Git. See
`STATUS.md` for the active phase and runnable commands.
