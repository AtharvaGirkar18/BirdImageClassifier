@echo off
REM Quick setup script for Windows
echo.
echo ========================================
echo  Bird Classifier - Quick Setup
echo ========================================
echo.

REM Check if Node.js is installed
where /q node
if errorlevel 1 (
    echo ERROR: Node.js not found!
    echo Please install from: https://nodejs.org/
    pause
    exit /b 1
)

echo [1/2] Installing frontend dependencies...
cd frontend
call npm install

if errorlevel 1 (
    echo ERROR: npm install failed
    pause
    exit /b 1
)

cd ..
echo.
echo [2/2] Setup complete!
echo.
echo ========================================
echo Next steps:
echo ========================================
echo.
echo Terminal 1 - Start Backend:
echo   C:\Users\handj\miniconda3\Scripts\activate.bat bird-img-gpu ^&^& python app.py
echo.
echo Terminal 2 - Start Frontend:
echo   cd frontend ^&^& npm run dev
echo.
echo Then open: http://localhost:3000
echo.
echo ========================================
echo.
pause
