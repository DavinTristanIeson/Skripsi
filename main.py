import logging
from fastapi import FastAPI
import server.controllers
import server.routes

from common.logger import RegisteredLogger
import common.ipc as ipc
import server.routes.api

app = FastAPI()
server.controllers.exceptions.register_error_handlers(app)

RegisteredLogger().configure(
  level=logging.INFO,
  terminal=True
)

# locker = ipc.tasks.IPCTaskLocker()
# locker.initialize(
#   channel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
#   backchannel=ipc.client.TOPIC2SERVER_IPC_CHANNEL
# )
# locker.listen()

app.include_router(server.routes.topics.router, prefix="/api/topics")
app.include_router(server.routes.api.router, prefix="/api")
                   