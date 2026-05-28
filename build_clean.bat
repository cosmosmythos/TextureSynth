@echo off
echo =======================================
echo Cleaning and rebuilding from scratch...
echo =======================================

if exist build (
    echo Removing existing build directory...
    rmdir /s /q build
)

echo.
echo Configuring CMake...
cmake -S . -B build

echo.
echo Building...
cmake --build build --config Release

echo.
echo Done!
