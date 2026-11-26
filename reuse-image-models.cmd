@echo off
rem Reuse picture/music model files from another FreeVideo install (hardlinks, no copies).
rem Usage: reuse-image-models.cmd "C:\path\to\other\FreeVideo-engine" [--dry-run]
"%~dp0envs\unified\Scripts\python.exe" "%~dp0scripts\reuse_image_models.py" %*
pause
exit /b %ERRORLEVEL%
