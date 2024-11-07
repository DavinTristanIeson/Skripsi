import asyncio
from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import server.controllers
import server.routes

from common.logger import RegisteredLogger
import common.ipc as ipc
import server.routes.evaluation

@asynccontextmanager
async def lifespan(app):
  try:
    yield
  except asyncio.exceptions.CancelledError:
    pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"]
)


is_app = os.getenv("APP")
RegisteredLogger().configure(
  level=logging.WARNING if is_app else logging.DEBUG,
  terminal=True
)

api_app = FastAPI(lifespan=lifespan)
api_app.include_router(server.routes.association.router, prefix="/projects")
api_app.include_router(server.routes.topics.router, prefix="/projects")
api_app.include_router(server.routes.projects.router, prefix="/projects")
api_app.include_router(server.routes.evaluation.router, prefix="/projects")
api_app.include_router(server.routes.general.router, prefix="")
api_app.include_router(server.routes.debug.router, prefix="/debug")
server.controllers.exceptions.register_error_handlers(api_app)

app.mount('/api', api_app)

if os.path.exists("views"):
  app.mount("/", StaticFiles(directory="views", html = True), name="static")
else:
  print("No interface files has been found in /views. You should run \"python scripts/download_interface.py\" to download the default interface.")


locker = ipc.taskqueue.IPCTaskClient()
locker.initialize(
  channel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
)