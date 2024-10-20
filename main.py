import atexit
import logging
import threading
from fastapi import FastAPI
import server.controllers
import server.routes

from common.logger import RegisteredLogger
import common.ipc as ipc

app = FastAPI()
server.controllers.exceptions.register_error_handlers(app)

RegisteredLogger().configure(
  level=logging.INFO,
  terminal=True
)

locker = ipc.taskqueue.IPCTaskLocker()
locker.initialize(
  channel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
  backchannel=ipc.client.TOPIC2SERVER_IPC_CHANNEL
)

stop_event = threading.Event()
locker.listen(stop_event)

@atexit.register
def cleanup():
  stop_event.set()

app.include_router(server.routes.topics.router, prefix="/api/topics")