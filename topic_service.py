import logging

import concurrent.futures
import atexit
import threading

import common.ipc as ipc
from common.ipc.client import IntraProcessCommunicator
from common.ipc.requests import IPCRequestType, IPCRequest
from common.ipc.responses import IPCResponse, IPCResponseData, IPCResponseStatus
from common.ipc.tasks import ipc_task_handler
from common.logger import RegisteredLogger

@ipc_task_handler
def placeholder(comm: IntraProcessCommunicator, message: IPCRequest):
  client = ipc.client.IPCClient(ipc.client.TOPIC2SERVER_IPC_CHANNEL)
  client.send(IPCResponse(
    id=message.id,
    data=IPCResponseData.Empty(),
    error=None,
    status=IPCResponseStatus.Success,
  ))
  print(message)
  yield IPCResponseData.Empty()


pool = concurrent.futures.ProcessPoolExecutor(4)
listener_pool = concurrent.futures.ThreadPoolExecutor(8)
receiver = ipc.tasks.IPCTaskReceiver()
receiver.initialize(
  channel=ipc.client.TOPIC2SERVER_IPC_CHANNEL,
  backchannel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
  handlers={
    IPCRequestType.TopicModeling: placeholder,
    IPCRequestType.TopicCorrelationPlot: placeholder,
    IPCRequestType.CreateTopic: placeholder,
    IPCRequestType.DeleteTopics: placeholder,
    IPCRequestType.MergeTopics: placeholder,
  },
  listener_pool=listener_pool,
  pool=pool
)

RegisteredLogger().configure(
  level=logging.INFO,
  terminal=True
)

receiver.listen()

@atexit.register
def cleanup():
  receiver.listener.running = False  


# Allows user to kill process with Ctrl + C
while True:
  input()
