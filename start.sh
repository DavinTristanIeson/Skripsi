#!/bin/bash
setx APP=1

# Activate virtual environment
source venv/bin/activate

# Run FastAPI with uvicorn
fastapi run