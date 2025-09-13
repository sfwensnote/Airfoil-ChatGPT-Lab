@echo off
title Airfoil Assistant Server
cd /d %~dp0
call venv\Scripts\activate

:: 启动后端
start "Backend" cmd /k uvicorn backend:app --host 0.0.0.0 --port 8000

:: 等待 3 秒确保后端起来
timeout /t 3 /nobreak >nul

:: 启动前端
start "Frontend" cmd /k streamlit run app.py --server.address 0.0.0.0 --server.port 8501
