@echo off
REM Activate virtual environment
call venv\Scripts\activate

REM Running FastAPI
fastapi run main.py

pause