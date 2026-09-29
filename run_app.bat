@echo off
title RiceTec Sales & Opportunity Intelligence Chatbot
echo ========================================================
echo   RiceTec Sales & Opportunity Intelligence Assistant
echo ========================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b 1
)

echo [1/2] Verifying and installing required packages...
python -m pip install -r requirements.txt --quiet

echo.
echo [2/2] Launching RiceTec Chatbot on your local system...
echo The application will open automatically in your default browser.
echo Press Ctrl+C in this terminal window to stop the application.
echo.
python -m streamlit run app.py
pause
