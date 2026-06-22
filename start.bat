@echo off
title Chat AXE - RAG Chatbot
echo.
echo  ============================
echo   Chat AXE - RAG Chatbot
echo  ============================
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Docker is not installed or not running.
    echo  Install Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo  [1/4] Starting PostgreSQL and Ollama via Docker...
docker-compose up -d db ollama
if errorlevel 1 (
    echo  [ERROR] docker-compose failed. Is Docker Desktop running?
    pause
    exit /b 1
)

echo  [2/4] Waiting for services to start...
timeout /t 10 /noq >nul

echo  [3/4] Pulling embedding model (skip if already present)...
REM Try local ollama first, then docker
where ollama >nul 2>&1
if %errorlevel%==0 (
    ollama pull nomic-embed-text
) else (
    for /f "tokens=*" %%i in ('docker-compose ps -q ollama') do (
        docker exec %%i ollama pull nomic-embed-text
    )
)

echo  [4/4] Starting Flask app...
echo.

REM Activate venv or create one
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo  Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo  Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo  ----------------------------------------
echo   Open http://localhost:5000 in browser
echo   Press Ctrl+C to stop
echo  ----------------------------------------
echo.

python app.py
