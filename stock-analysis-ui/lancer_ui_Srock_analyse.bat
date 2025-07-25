@echo off
cd /d "%~dp0src"
call ..\.venv\Scripts\activate
python -m ui.main_window
pause