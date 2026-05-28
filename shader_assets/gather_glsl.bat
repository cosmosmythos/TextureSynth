@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%all_glsl.txt"

if exist "%OUT%" del "%OUT%"

echo Gathering .glsl files from: "%ROOT%"
echo Output: "%OUT%"
echo.

for /r "%ROOT%" %%F in (*.glsl) do (
    echo ===== %%~fF =====>>"%OUT%"
    type "%%~fF" >>"%OUT%"
    echo.>>"%OUT%"
    echo Added: %%~fF
)

echo.
echo Done.
pause