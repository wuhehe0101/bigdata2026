@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYCMD=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYCMD=%~dp0..\.venv\Scripts\python.exe"
echo === 先执行数值单元测试 ===
"%PYCMD%" -m unittest discover -s tests -v
if errorlevel 1 (
    echo 数值测试失败，已中止。
    pause
    exit /b 1
)
echo === 开始生成结果（三组完整600秒情景） ===
"%PYCMD%" run_batch.py --sensitivity
if errorlevel 1 (
    echo 计算失败，请复制上面的错误日志。
) else (
    echo 完成！双击index.html查看计算网页。
)
pause
