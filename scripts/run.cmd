@echo off
rem Runs the ClinicalLingo CLI with the portable Python.  Examples:
rem   run explain "echogenic" --mode word
rem   run explain "The liver is hypoechoic relative to the renal cortex." --mode sentence --quiz
rem   run doctor
call "%~dp0env.cmd"
cd /d "%CL_REPO%"
"%CL_ROOT%\win\python\python.exe" "%CL_REPO%\cl.py" %*
