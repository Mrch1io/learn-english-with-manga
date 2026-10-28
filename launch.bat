@echo off
setlocal
call "%~dp0run.bat" --gui %*
if errorlevel 1 pause
endlocal
