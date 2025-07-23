@echo off
SET APP=1
REM Activate virtual environment
call venv\Scripts\activate

REM Running FastAPI
fastapi run main.py

pause