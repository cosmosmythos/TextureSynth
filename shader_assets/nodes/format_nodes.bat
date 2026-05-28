@echo off
setlocal
REM Usage: format_nodes.bat [folder]
REM   Default folder = current directory.

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=%CD%"

python "%~dp0format_nodes.py" "%TARGET%"
endlocal