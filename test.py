import multiprocessing
import time
from typing import Generic, TypeVar

import pydantic

T = TypeVar("T")
class TestPydantic(pydantic.BaseModel, Generic[T], ):
    a: T

class QueueWrapper:
    def __init__(self, queue):
        self.queue = queue

def child_process(wrapper: QueueWrapper, value: TestPydantic):
    print("[Child] Putting message in queue...")
    value.a *= 2.0
    wrapper.queue.put(value)

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    wrapper = QueueWrapper(queue)

    p = multiprocessing.Process(target=child_process, args=(wrapper, TestPydantic(a=5)))
    p.start()
    p.join()

    print("[Parent] Getting message from queue...")
    msg = wrapper.queue.get()
    print("[Parent] Received:", msg)
