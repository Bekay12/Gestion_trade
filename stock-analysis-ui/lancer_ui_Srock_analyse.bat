@echo off
REM Script d'autostart pour Stock Analysis UI
REM Active l'environnement virtuel et lance l'application

cd /d "C:\Users\berti\Desktop\Mes documents\Gestion_trade\stock-analysis-ui\src"

REM Activer l'environnement virtuel
call "..\. venv\Scripts\activate.bat"

REM Lancer l'application
python "ui\main_window.py"

REM Si l'application se ferme, garder la fenêtre ouverte
pause
