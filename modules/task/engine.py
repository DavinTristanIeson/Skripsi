from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

from modules.baseclass import Singleton
from modules.logger.provisioner import ProvisionedLogger
from modules.task.responses import TaskResponse

scheduler = AsyncIOScheduler(
  jobstores=dict(default=MemoryJobStore()),
  executors=dict(default=ThreadPoolExecutor(max_workers=1)),
)

# Register apscheduler logger
ProvisionedLogger().provision("apscheduler")

__all__ = [
  "scheduler"
]