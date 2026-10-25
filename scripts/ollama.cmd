@echo off
rem Runs the portable ollama client with the drive's environment, e.g.  ollama list  /  ollama pull phi3:mini
call "%~dp0env.cmd"
set "USERPROFILE=%CL_ROOT%\home"
set "HOME=%CL_ROOT%\home"
"%CL_ROOT%\win\ollama\ollama.exe" %*
