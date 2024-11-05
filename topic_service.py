import logging
import os
import threading
from common.logger import RegisteredLogger

is_app = os.getenv("APP")

RegisteredLogger().configure(
  level=logging.WARNING if is_app else logging.DEBUG,
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


  pool = concurrent.futures.ThreadPoolExecutor(16)
  receiver = ipc.taskqueue.IPCTaskServer()
  receiver.initialize(
    channel=ipc.client.TOPIC2SERVER_IPC_CHANNEL,
    backchannel=ipc.client.SERVER2TOPIC_IPC_CHANNEL,
    handlers={
      IPCRequestType.TopicModeling: topic.controllers.model.topic_modeling,
      IPCRequestType.TopicSimilarity: topic.controllers.plots.topic_similarity_plot,
      IPCRequestType.Topics: topic.controllers.plots.topic_plot,
      IPCRequestType.Association: topic.controllers.association.association_plot,
      IPCRequestType.Evaluation: topic.controllers.evaluation.evaluate_topics,
      IPCRequestType.CreateTopic: placeholder,
      IPCRequestType.DeleteTopics: placeholder,
      IPCRequestType.MergeTopics: placeholder,
    },
    pool=pool
  )

  stop_event = threading.Event()
  receiver.listen(stop_event)

  @atexit.register
  def cleanup():
    stop_event.set()
    pool.shutdown(wait=False, cancel_futures=True)

  # Allows user to kill process with Ctrl + C
  while True:
    input()
