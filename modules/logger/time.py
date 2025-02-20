import logging
import time
from typing import Optional, Union

from .provisioner import ProvisionedLogger


class TimeLogger:
  start_time: float
  title: str
  logger: logging.Logger
  report_start: bool
  def __init__(self, logger: Union[str, logging.Logger], title: str, *, report_start: bool = False) -> None:
    if isinstance(logger, logging.Logger):
      self.logger = logger
    else:
      self.logger = ProvisionedLogger().provision(logger)
    self.title = title
    self.report_start = report_start

  def derive(self, title: Optional[str] = None, *, report_start: Optional[bool] = None)->"TimeLogger":
    logger = TimeLogger(
      self.logger.name,
      title=self.title,
      report_start=self.report_start
    )

    if title:
      logger.title = title

    if report_start:
      logger.report_start = report_start
    
    return logger
    
  def __enter__(self):
    if self.report_start:
      self.logger.debug(f"{self.title} - START")
    self.start_time = time.perf_counter()

  def __exit__(self, *args):
    elapsed_time = time.perf_counter() - self.start_time
    if elapsed_time < 1e-6:
      elapsed_time *= 1e6
      unit = "μs"
    elif elapsed_time < 1e-3:
      elapsed_time *= 1e3
      unit = "ms"
    else:
      unit = "s"
    self.logger.info(f"{self.title} - {elapsed_time:.4f} {unit}")

__all__ = [
  "TimeLogger"
]