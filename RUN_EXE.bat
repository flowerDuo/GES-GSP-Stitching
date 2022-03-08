@echo off

set target=''
set file=F:\Projects\C++\GES_Stitching\RUN_FILE.txt
setlocal enabledelayedexpansion
for /f %%i in (%file%) do (
set target=%%i
if "%target%"=="" ( echo Input is empty!---------------------------------) else (
echo Start !target! !---------------------------------
F:\Projects\C++\GES_Stitching\GES_Stitching.exe !target! )
echo Finish !target! !---------------------------------
)

echo All finish !---------------------------------
pause