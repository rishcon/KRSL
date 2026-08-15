@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :missing_python
if not exist "data\processed\holistic-v1" goto :missing_artifacts

echo Building hand-centric training-sequence-v3...
".venv\Scripts\python.exe" "scripts\build_training_sequences.py" --manifest "data\manifests\krsl20_v1.csv" --artifact-root "data\processed\holistic-v1" --output-root "data\interim\training-sequence-v3" --report "reports\processing\training-sequence-v3_summary.json" --hand-centric
if errorlevel 1 goto :failed

echo.
echo Training hand-centric BiLSTM v3...
".venv\Scripts\python.exe" "scripts\train_lstm_baseline.py" --manifest "data\manifests\krsl20_v1.csv" --sequence-root "data\interim\training-sequence-v3" --report-dir "reports\experiments\lstm-handcentric-v3" --epochs 30 --batch-size 32 --model-type "hand-centric-bilstm-v3"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%

:missing_python
echo ERROR: .venv\Scripts\python.exe was not found.
goto :failed

:missing_artifacts
echo ERROR: data\processed\holistic-v1 was not found.
goto :failed

:failed
pause
exit /b 1
