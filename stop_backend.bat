@echo off
echo Stopping Backend on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    taskkill /f /pid %%a
)
echo Backend stopped successfully.
pause
