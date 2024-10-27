import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import server.controllers
import server.routes

from common.logger import RegisteredLogger
import common.ipc as ipc

app = FastAPI()
server.controllers.exceptions.register_error_handlers(app)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"]
)


RegisteredLogger().configure(
  level=logging.DEBUG,
  terminal=True
)

app.include_router(server.routes.topics.router, prefix="/api/projects")
app.include_router(server.routes.projects.router, prefix="/api/projects")
app.include_router(server.routes.general.router, prefix="/api")
app.include_router(server.routes.debug.router, prefix="/api/debug")

locker = ipc.taskqueue.IPCTaskClient()
locker.initialize(
  channel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
)