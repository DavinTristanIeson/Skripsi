import logging
import multiprocessing
from common.logger import RegisteredLogger

RegisteredLogger().configure(
  level=logging.DEBUG,
  terminal=True
)
if __name__ == "__main__":
  import concurrent.futures
  import atexit

  import common.ipc as ipc
  from common.ipc.requests import IPCRequestType
  from common.ipc.responses import IPCResponseData
  from common.ipc.task import IPCTask

  import topic.controllers

  def placeholder(comm: IPCTask):
    print(comm.request)
    comm.success(IPCResponseData.Empty(), "Placeholder")


  pool = concurrent.futures.ProcessPoolExecutor(4)
  listener_pool = concurrent.futures.ThreadPoolExecutor(8)
  lock = multiprocessing.Lock()
  receiver = ipc.taskqueue.IPCTaskReceiver()
  receiver.initialize(
    channel=ipc.client.TOPIC2SERVER_IPC_CHANNEL,
    backchannel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
    lock=lock,
    handlers={
      IPCRequestType.TopicModeling: topic.controllers.model.topic_modeling,
      IPCRequestType.TopicCorrelationPlot: placeholder,
      IPCRequestType.CreateTopic: placeholder,
      IPCRequestType.DeleteTopics: placeholder,
      IPCRequestType.MergeTopics: placeholder,
    },
    listener_pool=listener_pool,
    pool=pool
  )

  receiver.listen()

  @atexit.register
  def cleanup():
    receiver.listener.running = False
    pool.shutdown(wait=False, cancel_futures=True)
    listener_pool.shutdown(wait=False, cancel_futures=True)

  # Allows user to kill process with Ctrl + C
  while True:
    input()
