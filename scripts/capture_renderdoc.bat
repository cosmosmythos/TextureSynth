@echo off
REM Thin wrapper around renderdoccmd.exe for capturing GPU frames.
REM
REM Usage:
REM   capture_renderdoc.bat                       -- captures engine_tests.exe
REM   capture_renderdoc.bat viewer                -- captures the standalone viewer
REM   capture_renderdoc.bat viewer --num-frames=3 -- captures 3 frames of the viewer
REM
REM The captured .rdc is written to .\captures\engine_capture.rdc (or
REM .\captures\viewer_capture.rdc). Open it in qrenderdoc.exe.
REM
REM The viewer has an in-process capture button (see RenderDocCapture.*
REM in src/viewer/), so for interactive use, launch qrenderdoc.exe,
REM attach to viewer.exe, and click "Capture Frame" in the UI. This
REM script is for headless / scripted captures.

setlocal
set "TARGET=engine_tests"
set "OUTNAME=engine_capture"
set "EXTRA_FLAGS="

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="engine" (
    set "TARGET=engine_tests"
    set "OUTNAME=engine_capture"
) else if /i "%~1"=="viewer" (
    set "TARGET=viewer"
    set "OUTNAME=viewer_capture"
) else (
    set "EXTRA_FLAGS=%EXTRA_FLAGS% %~1"
)
shift
goto parse_args

:done_parsing

if not exist "%~dp0..\captures" mkdir "%~dp0..\captures"

"C:\Program Files\RenderDoc\renderdoccmd.exe" capture ^
    --working-dir "%~dp0.." ^
    --capture-file "%~dp0..\captures\%OUTNAME%" ^
    --num-frames=1 %EXTRA_FLAGS% ^
    "%~dp0..\build\Release\%TARGET%.exe"

endlocal
