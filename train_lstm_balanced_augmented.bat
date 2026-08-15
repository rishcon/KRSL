@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :missing_python
if not exist "data\interim\training-sequence-v2" goto :missing_sequences

echo Training balanced and augmented velocity BiLSTM...
".venv\Scripts\python.exe" "scripts\train_lstm_baseline.py" --manifest "data\manifests\krsl20_v1.csv" --sequence-root "data\interim\training-sequence-v2" --report-dir "reports\experiments\lstm-balanced-augmented-v1" --epochs 30 --batch-size 32 --balanced-loss --augment
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%

:missing_python
echo ERROR: .venv\Scripts\python.exe was not found.
goto :failed

:missing_sequences
echo ERROR: data\interim\training-sequence-v2 was not found.
goto :failed

:failed
pause
exit /b 1
