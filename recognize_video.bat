@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Ошибка: не найдено окружение .venv
  echo Сначала запустите: python -m pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "models\holistic_landmarker.task" (
  echo Ошибка: не найден models\holistic_landmarker.task
  pause
  exit /b 1
)

if not exist "models\lstm-velocity-v1.pt" (
  echo Ошибка: не найдена модель models\lstm-velocity-v1.pt
  pause
  exit /b 1
)

set "VIDEO_PATH=%~1"
if not defined VIDEO_PATH set /p "VIDEO_PATH=Введите путь к видео: "
set "VIDEO_PATH=%VIDEO_PATH:"=%"

if not exist "%VIDEO_PATH%" (
  echo Ошибка: видео не найдено: %VIDEO_PATH%
  pause
  exit /b 1
)

echo.
echo Обрабатываю видео...
.venv\Scripts\python.exe scripts\recognize_video.py --video "%VIDEO_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
