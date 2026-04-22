@echo off
title OBJ SCANNER - ALL-IN-ONE
cd /d "G:\OBJScanner\backend"

echo ==========================================
echo   OBJ SCANNER - ALL-IN-ONE START
echo ==========================================

echo Preparing Universal AR Object Scanner Backend...
echo Starting Backend Server...

:: Start the backend and open the browser after a short delay
start /b python main.py
timeout /t 5 >nul
echo Opening Frontend...
start http://127.0.0.1:8000

echo.
echo Server is running! 
echo Check the terminal below for your Mobile URL.
echo ==========================================
pause
