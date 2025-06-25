@echo off
rem Activate virtual environment and launch GPOST
call "%~dp0venv\Scripts\activate.bat"
python translate.py

