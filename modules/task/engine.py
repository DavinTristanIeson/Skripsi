from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

from modules.logger.provisioner import ProvisionedLogger

topic_modeling_job_store = MemoryJobStore()
scheduler = AsyncIOScheduler(
  executors=dict(
    default=ThreadPoolExecutor(4)
  ),
  jobstores=dict(
    default=MemoryJobStore(),
    topic_modeling=topic_modeling_job_store
  ),
)

# Register apscheduler logger
ProvisionedLogger().provision("apscheduler")

__all__ = [
  "scheduler",
  "topic_modeling_job_store"
]