import logging
import common.controllers.ipc as ipc
import threading

from common.logger import RegisteredLogger

logger = logging.getLogger("Topic Modeling")
def placeholder(message: ipc.IPCMessage):
  print(message)

listener = ipc.IPCListener({
  ipc.IPCMessageType.TopicModelingRequest: placeholder,
  ipc.IPCMessageType.TopicHierarchyPlotRequest: placeholder,
})

RegisteredLogger().configure(
  level=logging.INFO,
  terminal=True
)

thread = threading.Thread(target=listener.listen, daemon=True)
thread.start()

# Allows user to kill process with Ctrl + C
while True:
  input()