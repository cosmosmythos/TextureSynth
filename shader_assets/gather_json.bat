@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%all_json.txt"

if exist "%OUT%" del "%OUT%"

echo Gathering .json files from: "%ROOT%"
echo Output: "%OUT%"
echo.

for /r "%ROOT%" %%F in (*.json) do (
    echo ===== %%~fF =====>>"%OUT%"
    type "%%~fF" >>"%OUT%"
    echo.>>"%OUT%"
    echo Added: %%~fF
)

echo.
echo Done.
pause