import logging

import concurrent.futures
import threading

import common.ipc as ipc
from common.ipc.requests import IPCRequestType, IPCRequest
from common.ipc.responses import IPCResponse, IPCResponseData, IPCResponseStatus
from common.logger import RegisteredLogger

def placeholder(message: IPCRequest):
  client = ipc.client.IPCClient(ipc.client.TOPIC2SERVER_IPC_CHANNEL)
  client.send(IPCResponse(
    id=message.id,
    data=IPCResponseData.Empty(),
    error=None,
    status=IPCResponseStatus.Success,
  ))
  print(message)


pool = concurrent.futures.ProcessPoolExecutor(4)
receiver = ipc.tasks.IPCTaskReceiver()
receiver.initialize(
  backchannel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
  handlers={
    IPCRequestType.TopicModeling: placeholder,
    IPCRequestType.TopicCorrelationPlot: placeholder,
    IPCRequestType.CreateTopic: placeholder,
    IPCRequestType.DeleteTopics: placeholder,
    IPCRequestType.MergeTopics: placeholder,
  },
  pool=pool
)

RegisteredLogger().configure(
  level=logging.INFO,
  terminal=True
)

receiver.listen()
# Allows user to kill process with Ctrl + C
while True:
  input()