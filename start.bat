@echo off
REM Airfoil Lab 一键启动脚本 (Windows)
REM 同时启动后端 (FastAPI) 和前端 (Next.js)

echo 🚀 Starting Airfoil Lab...

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM 启动后端
echo 📡 Starting backend (port 8000)...
start "Airfoil Backend" cmd /c "python -m uvicorn backend:app --host 0.0.0.0 --port 8000"

REM 等待后端启动
timeout /t 3 /nobreak > nul

REM 启动前端
echo 🌐 Starting frontend (port 3000)...
cd airfoil-lab-react
start "Airfoil Frontend" cmd /c "npm run dev"

REM 等待前端启动
timeout /t 3 /nobreak > nul

echo.
echo ✅ Airfoil Lab is running!
echo    Frontend: http://localhost:3000
echo    Backend:  http://localhost:8000
echo.
echo Close the terminal windows to stop the services.
pause
