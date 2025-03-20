import asyncio
from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from modules.api.wrapper import ApiErrorResult
import routes

from modules.logger import ProvisionedLogger
from modules.api import register_error_handlers
from modules.task import scheduler

@asynccontextmanager
async def lifespan(app):
  try:
    scheduler.start()
    yield
  except asyncio.exceptions.CancelledError:
    pass
  scheduler.shutdown()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"]
)

is_app = os.getenv("APP")
ProvisionedLogger().configure(
  level=logging.WARNING if is_app else logging.DEBUG,
  terminal=True
)

api_app = FastAPI(lifespan=lifespan, responses={
  400: dict(model=ApiErrorResult),
  403: dict(model=ApiErrorResult),
  404: dict(model=ApiErrorResult),
  422: dict(model=ApiErrorResult),
  500: dict(model=ApiErrorResult),
})
api_app.include_router(routes.project.router, prefix="/projects")
api_app.include_router(routes.general.router, prefix="")
api_app.include_router(routes.table.router, prefix="/table/{project_id}")
api_app.include_router(routes.topic.router, prefix="/topics/{project_id}")
register_error_handlers(api_app)

app.mount('/api', api_app)

if os.path.exists("views"):
  app.mount("/", StaticFiles(directory="views", html = True), name="static")
else:
  print("No interface files has been found in /views. You should run \"python scripts/download_interface.py\" to download the default interface.")
