@echo off
@echo off
cd /d %~dp0

for /f "delims=" %%i in ('findstr __version__ bt\__version__.py') do set TARGET_VERSION=%%i
set TARGET_VERSION=%TARGET_VERSION:~15,-1%

for /f "delims=" %%i in ('pip show bt 2^>nul ^| findstr /B /C:"Version:"') do set INSTALLED_VERSION=%%i
set INSTALLED_VERSION=%INSTALLED_VERSION:~9%

echo Target version : %TARGET_VERSION%
echo Installed version : %INSTALLED_VERSION%

if "%INSTALLED_VERSION%"=="%TARGET_VERSION%" (
    echo [✓] bt %INSTALLED_VERSION% already installed. Skipping reinstall.
) else (
    echo [!] Installing bt %TARGET_VERSION%...
    pip uninstall -y bt
    pip install -e . --use-pep517
)

echo [OK] Development environment is ready!
pause
