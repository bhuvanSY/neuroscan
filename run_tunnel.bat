@echo off
setlocal enabledelayedexpansion

title NeuroScan AI - Live GPU Cloudflare Tunnel
echo ===============================================================================
echo            NEUROSCAN AI - SECURE HTTPS CLOUDFLARE TUNNEL (GPU BRIDGE)
echo ===============================================================================
echo.
echo Connecting your local RTX 4050 GPU backend (port 8000) to the public internet...
echo This generates a secure HTTPS URL for Vercel and mobile devices.
echo.

if not exist "%~dp0tools\cloudflared.exe" (
    echo [INFO] Downloading cloudflared.exe...
    if not exist "%~dp0tools" mkdir "%~dp0tools"
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '%~dp0tools\cloudflared.exe'"
)

echo [READY] Launching Cloudflare Tunnel with HTTP/2 protocol...
echo Once you see "Your quick Tunnel has been created! Visit it at: https://xxxx.trycloudflare.com",
echo COPY THAT HTTPS URL and paste it into the "Connect Real GPU Backend" settings modal in Vercel!
echo.
echo Press Ctrl+C at any time to close the tunnel.
echo ===============================================================================
echo.

"%~dp0tools\cloudflared.exe" tunnel --protocol http2 --edge-ip-version 4 --url http://127.0.0.1:8000

pause
