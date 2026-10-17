@echo off
title NeuroScan AI - FastAPI Backend Service (RTX 4050 GPU)
echo ==========================================================
echo Starting NeuroScan AI Backend Service on http://127.0.0.1:8000
echo ==========================================================
cd /d "%~dp0"
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
pause
