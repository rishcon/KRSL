@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :missing_python
if not exist "models\holistic_landmarker.task" goto :missing_holistic
if not exist "models\lstm-handcentric-v3.pt" goto :missing_checkpoint

set "VIDEO_PATH=%~1"
if defined VIDEO_PATH goto :video_ready
set /p "VIDEO_PATH=Enter the full path to the video: "

:video_ready
if not defined VIDEO_PATH goto :missing_video
set "VIDEO_PATH=%VIDEO_PATH:"=%"
if not exist "%VIDEO_PATH%" goto :missing_video

echo.
echo Processing video...
".venv\Scripts\python.exe" "scripts\recognize_video.py" --video "%VIDEO_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%

:missing_python
echo ERROR: .venv\Scripts\python.exe was not found.
goto :failed

:missing_holistic
echo ERROR: models\holistic_landmarker.task was not found.
goto :failed

:missing_checkpoint
echo ERROR: models\lstm-handcentric-v3.pt was not found.
goto :failed

:missing_video
echo ERROR: The video path is empty or the file does not exist.
goto :failed

:failed
pause
exit /b 1
