@echo off
rem ClinicalLingo portable environment (Windows).
rem Every path is relative to the folder this file lives in, so the drive letter can change.
rem Expected layout:  <root>\ClinicalLingo (repo)  <root>\win\python  <root>\win\ollama  <root>\models
set "CL_ROOT=%~dp0"
if "%CL_ROOT:~-1%"=="\" set "CL_ROOT=%CL_ROOT:~0,-1%"
set "CL_REPO=%CL_ROOT%\ClinicalLingo"
set "OLLAMA_MODELS=%CL_ROOT%\models"
set "OLLAMA_HOST=127.0.0.1:11434"
set "OLLAMA_KEEP_ALIVE=30m"
set "OLLAMA_TMPDIR=%CL_ROOT%\tmp"
set "PIP_CACHE_DIR=%CL_ROOT%\pip-cache"
set "PATH=%CL_ROOT%\win\ollama;%CL_ROOT%\win\python;%CL_ROOT%\win\python\Scripts;%PATH%"
if not exist "%CL_ROOT%\tmp" mkdir "%CL_ROOT%\tmp"
