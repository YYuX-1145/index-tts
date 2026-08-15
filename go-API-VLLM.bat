@echo off
set FFMPEG_PATH=%CD%\ffmpeg\bin
set PYTHON_PATH=%CD%\indextts2runtime\
set SC_PATH=%CD%\indextts2runtime\Scripts
set HF_HOME=%CD%\checkpoints
set PATH=%FFMPEG_PATH%;%SC_PATH%;%PYTHON_PATH%;%PATH%

set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
for /d %%i in ("%VS_PATH%\VC\Tools\MSVC\*") do (
    set MSVC_VER=%%~nxi
    goto :found
)

:found
if not defined MSVC_VER (
    echo [WARNING]  MSVC NOT FOUND
    goto :run
)
set PATH=%PATH%;%VS_PATH%\VC\Tools\MSVC\%MSVC_VER%\bin\Hostx64\x64
echo MSVC VER: %MSVC_VER%

:run
indextts2runtime\python.exe indextts2_api.py --version 2.5 --fp16 --gpu_memory_utilization 0.25
pause
