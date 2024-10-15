from fastapi import FastAPI
import controllers.server
import routes.server

app = FastAPI()
controllers.server.exceptions.register_error_handlers(app)

app.include_router(routes.server.topics.router, prefix="/api/topics")