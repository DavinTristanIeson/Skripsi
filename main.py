import logging
from fastapi import FastAPI
import server.controllers
import server.routes
from common.logger import RegisteredLogger

app = FastAPI()
server.controllers.exceptions.register_error_handlers(app)

RegisteredLogger().configure(
  level=logging.INFO,
  terminal=True
)

app.include_router(server.routes.topics.router, prefix="/api/topics")