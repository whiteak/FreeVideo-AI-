@echo off
rem Bring this fork up to date with upstream Eclipse, only when you choose.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update_from_upstream.ps1" %*
pause
exit /b %ERRORLEVEL%
