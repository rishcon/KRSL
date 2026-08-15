@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" goto :missing_python
if not exist "models\holistic_landmarker.task" goto :missing_holistic
if not exist "models\lstm-handcentric-v3.pt" goto :missing_checkpoint

start "" ".venv\Scripts\pythonw.exe" "scripts\run_gui.py"
exit /b 0

:missing_python
echo ERROR: .venv\Scripts\pythonw.exe was not found.
goto :failed

:missing_holistic
echo ERROR: models\holistic_landmarker.task was not found.
goto :failed

:missing_checkpoint
echo ERROR: models\lstm-handcentric-v3.pt was not found.
goto :failed

:failed
pause
exit /b 1
