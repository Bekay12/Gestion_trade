@echo off
REM Wrapper lance par le Planificateur de taches Windows.
REM Le planificateur n'active pas le venv : on appelle son python en absolu.
REM Toute sortie est journalisee, sinon une panne serait silencieuse.

set "ROOT=%~dp0.."
set "FINVIZ_STATE=%ROOT%\agent\finviz_state.json"

echo. >> "%ROOT%\agent\finviz_watch.log"
echo ===== %DATE% %TIME% ===== >> "%ROOT%\agent\finviz_watch.log"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\agent\Finviz_watch.py" >> "%ROOT%\agent\finviz_watch.log" 2>&1
echo [exit %ERRORLEVEL%] >> "%ROOT%\agent\finviz_watch.log"
