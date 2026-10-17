@echo off
title NeuroScan AI Launcher
echo ==========================================================
echo Launching NeuroScan AI (Backend + Frontend)
echo ==========================================================
cd /d "%~dp0"

echo Starting FastAPI Backend (CUDA RTX 4050)...
start "NeuroScan AI Backend" cmd /k "python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting React Vite Frontend...
start "NeuroScan AI Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 2 /nobreak >nul
start http://localhost:3000

echo System running at http://localhost:3000
