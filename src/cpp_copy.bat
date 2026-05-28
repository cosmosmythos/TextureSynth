@echo off
setlocal enabledelayedexpansion
set "OUT=all_cpp_hpp.txt"
set "ROOT=%cd%"

rem Start fresh each run
> "%OUT%" (
    echo ===== Combined source dump =====
    echo Folder: %ROOT%
    echo.
)

for /r "%ROOT%" %%F in (*.cpp) do call :AddFile "%%~fF"
for /r "%ROOT%" %%F in (*.hpp) do call :AddFile "%%~fF"

echo Done. Created "%OUT%"
pause
exit /b

:AddFile
set "FILE=%~1"

rem Skip anything inside a folder named "viewer"
rem String substitution: if replacing \viewer\ changes the path, it was in there
set "STRIPPED=!FILE:\viewer\=!"
if /I "!STRIPPED!" neq "!FILE!" exit /b

>> "%OUT%" (
    echo ==================================================
    echo FILE: !FILE!
    echo ==================================================
    type "!FILE!"
    echo.
    echo.
)
exit /b