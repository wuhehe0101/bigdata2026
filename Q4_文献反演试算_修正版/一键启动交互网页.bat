@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYCMD=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYCMD=%~dp0..\.venv\Scripts\python.exe"
echo === 吉隆826 Streamlit 动态模拟 ===
echo Python: %PYCMD%
echo 提示：第一次运行请先执行 python -m pip install -r requirements.txt
echo 请等待出现 http://localhost:8501 后在浏览器中打开。
"%PYCMD%" -m streamlit run app.py
if errorlevel 1 (
    echo 启动未成功。确认已在当前Python环境安装streamlit及requirements依赖。
    pause
)
