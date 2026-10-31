@echo off
chcp 65001 >nul
echo ============================================
echo   Raise GPU power limit for RTX 5070 Ti Laptop
echo   Current: 70W   Max allowed: 140W
echo   Requires Administrator privileges.
echo ============================================
echo.
nvidia-smi -q -d POWER | findstr /C:"Current Power Limit" /C:"Max Power Limit"
echo.
echo Setting power limit to 140W ...
nvidia-smi -pl 140
if errorlevel 1 (
    echo.
    echo [FAILED] This laptop controls GPU TGP from the EC/vBIOS.
    echo          Use the keyboard hotkey Fn+Q instead:
    echo            press it until the power LED turns RED
    echo            ^(Quiet -^> Balanced -^> Performance/Beast^)
    echo          Or open LegionZone and pick Performance mode.
) else (
    echo.
    echo [OK] Done. Verify with:
    echo      nvidia-smi -q -d POWER
)
echo.
pause
