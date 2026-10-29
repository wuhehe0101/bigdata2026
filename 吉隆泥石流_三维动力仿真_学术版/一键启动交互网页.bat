@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
echo Launching Streamlit from %CD%
"%PYTHON_EXE%" -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo Install dependencies with: "%PYTHON_EXE%" -m pip install -r requirements.txt
  pause
)
