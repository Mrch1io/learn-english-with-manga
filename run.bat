@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
set "MANGA_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "MANGA_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined MANGA_PYTHON if exist "%~dp0.gpu-python" set /p "MANGA_PYTHON="<"%~dp0.gpu-python"
if not defined MANGA_PYTHON (
    echo No project GPU Python is configured. See README.md for setup.
    exit /b 1
)
if not exist "%MANGA_PYTHON%" (
    echo Configured GPU Python was not found: "%MANGA_PYTHON%"
    exit /b 1
)
"%MANGA_PYTHON%" -c "import sys, torch; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0)) if ok else print('CUDA is unavailable. This project requires a CUDA GPU; CPU execution is disabled.'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo GPU check failed. Install CUDA-enabled PyTorch in the configured environment.
    exit /b 1
)
"%MANGA_PYTHON%" "%~dp0main.py" %*
exit /b %errorlevel%
