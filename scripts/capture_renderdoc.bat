@echo off
REM Adjust the path to your RenderDoc install
"C:\Program Files\RenderDoc\renderdoccmd.exe" capture ^
    --working-dir "%~dp0.." ^
    --capture-file "%~dp0..\captures\engine_capture" ^
    "%~dp0..\build\tests\Release\engine_tests.exe"
