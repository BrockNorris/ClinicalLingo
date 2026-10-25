@echo off
rem Starts the portable Ollama server. Leave this window open while using ClinicalLingo.
rem USERPROFILE/HOME are redirected so Ollama's identity key is written to <root>\home, not C:\.
call "%~dp0env.cmd"
set "USERPROFILE=%CL_ROOT%\home"
set "HOME=%CL_ROOT%\home"
if not exist "%CL_ROOT%\home" mkdir "%CL_ROOT%\home"
echo ClinicalLingo: starting Ollama %CL_ROOT%\win\ollama  (models: %OLLAMA_MODELS%)
echo First model load from a flash drive can take 1-3 minutes; later requests are fast.
"%CL_ROOT%\win\ollama\ollama.exe" serve
