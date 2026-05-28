@echo off
echo =======================================
echo Building app (fast)...
echo =======================================

if not exist build (
    echo Build directory not found. Please run build_clean.bat first!
    exit /b 1
)

cmake -S . -B build >nul 2>&1
cmake --build build --config Release

echo.
echo Done!
