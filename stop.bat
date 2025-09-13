@echo off
title Stop Airfoil Assistant Server
echo Stopping Airfoil Assistant...

:: 强制关闭 uvicorn 和 streamlit 进程
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Backend"
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Frontend"

echo Done.
pause
