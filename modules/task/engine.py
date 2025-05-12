from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

from modules.logger.provisioner import ProvisionedLogger

scheduler = AsyncIOScheduler(
  executors=dict(
    # 2 subprocesses to run tasks in parallel.
    default=ProcessPoolExecutor(
      max_workers=2,
    )
  ),
  jobstores=dict(
    default=MemoryJobStore(),
  ),
)

# Register apscheduler logger
ProvisionedLogger().provision("apscheduler")

__all__ = [
  "scheduler",
]