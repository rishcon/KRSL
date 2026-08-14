# Phase 0 implementation plan

## Scope

Create the clean, testable project foundation and document the legacy reference.
No raw data, preprocessing, model training, or API code changes are in scope.

## Steps

1. Create the Python package, dependency metadata, test configuration, and
   lint/format configuration.
2. Exclude raw biometric data, generated features, model artifacts, and local
   secrets from Git.
3. Audit the legacy reference implementation and record compatibility facts.
4. Run formatting, linting, and tests; then record results in `STATUS.md`.

## Risks

- The legacy project may not run in a supported environment; it remains an
  isolated reference and will not be imported by production code.
- The local dataset directories are large and include biometric material, so
  accidental commits are a privacy and repository-size risk.

## Validation

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```
