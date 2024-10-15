import logging
import sys
import time
from typing import Optional


DEFAULT_STREAM_HANDLER = logging.StreamHandler(sys.stdout)
DEFAULT_STREAM_HANDLER.setFormatter(logging.Formatter('\033[38;5;247m%(asctime)s %(levelname)s\033[0m \033[1m[%(name)s]\033[0m: %(message)s'))

class TimeLogger:
  start_time: float
  title: str
  logger: logging.Logger
  report_start: bool
  def __init__(self, log_name: str, title: str, *, report_start: bool = False) -> None:
    self.logger = logging.getLogger(log_name)
    self.logger.setLevel(logging.INFO)
    self.logger.addHandler(DEFAULT_STREAM_HANDLER)
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
      self.logger.info(f"{self.title} - START")
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
