@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\build_training_sequences.py --manifest data\manifests\krsl20_v1.csv --artifact-root data\processed\holistic-v1 --output-root data\interim\training-sequence-v2 --report reports\processing\training-sequence-v2_summary.json --include-velocity
pause
