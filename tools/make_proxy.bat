@echo off
if "%~1"=="" (
    echo 用法：把视频文件拖到本 bat 图标上
    pause
    exit /b
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_proxy.ps1" -InputVideo "%~1"
pause