import asyncio
from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
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
server.controllers.exceptions.register_error_handlers(app)

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

app.include_router(server.routes.association.router, prefix="/api/projects")
app.include_router(server.routes.topics.router, prefix="/api/projects")
app.include_router(server.routes.projects.router, prefix="/api/projects")
app.include_router(server.routes.evaluation.router, prefix="/api/projects")
app.include_router(server.routes.general.router, prefix="/api")
app.include_router(server.routes.debug.router, prefix="/api/debug")

locker = ipc.taskqueue.IPCTaskClient()
locker.initialize(
  channel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
)