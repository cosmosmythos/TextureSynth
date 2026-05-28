@echo off
setlocal

set "OUT=all_python_files.txt"

> "%OUT%" (
    echo ===== Combined Python source dump =====
    echo Folder: %cd%
    echo.
)

for /r %%F in (*.py) do (
    >> "%OUT%" (
        echo ==================================================
        echo FILE: %%~fF
        echo ==================================================
        type "%%F"
        echo.
        echo.
    )
)

echo Done. Created "%OUT%"
pause