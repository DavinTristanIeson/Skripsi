import logging
import controllers.ipc as ipc
import threading

logger = logging.getLogger("Topic Modeling")
def placeholder(message: ipc.IPCMessage):
  print(message)

listener = ipc.IPCListener({
  ipc.IPCMessageType.TopicModelingRequest: placeholder,
  ipc.IPCMessageType.TopicHierarchyPlotRequest: placeholder,
})

thread = threading.Thread(target=listener.listen, daemon=True)
thread.start()

# Allows user to kill process with Ctrl + C
while True:
  input()