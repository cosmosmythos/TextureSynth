@echo off
setlocal

:: Find Blender's embedded Python.
set "BLENDER_PYTHON=C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe"

if not exist "%BLENDER_PYTHON%" (
    echo [ERROR] Blender Python not found at %BLENDER_PYTHON%
    echo Please update run_tests.bat if Blender is installed elsewhere.
    exit /b 1
)

echo [INFO] Using Python: %BLENDER_PYTHON%
echo [INFO] Running TextureSynth Automated Test Suite...
echo.

set "PYTHONPATH=%cd%"
"%BLENDER_PYTHON%" -m unittest discover tests -v

if %errorlevel% neq 0 (
    echo.
    echo [FAIL] Some tests failed!
    exit /b %errorlevel%
) else (
    echo.
    echo [SUCCESS] All tests passed!
)
