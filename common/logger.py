import logging
import sys
import time
from typing import ClassVar, Optional

from common.models.metaclass import Singleton


TERMINAL_STREAM_HANDLER = logging.StreamHandler(sys.stdout)
TERMINAL_STREAM_HANDLER.setFormatter(logging.Formatter('\033[38;5;247m%(asctime)s %(levelname)s\033[0m \033[1m[%(name)s]\033[0m: %(message)s'))

class RegisteredLogger(metaclass=Singleton):
  __logger_names: set[str]
  level: int = logging.INFO
  terminal: bool = False

  def __init__(self):
    super().__init__()
    self.__logger_names = set()
  
  def configure_logger(self, logger: logging.Logger):
    logger.setLevel(self.level)
    if self.terminal:
      logger.addHandler(TERMINAL_STREAM_HANDLER)
    else:
      logger.removeHandler(TERMINAL_STREAM_HANDLER)
  

  def provision(self, name: str)->logging.Logger:
    logger = logging.getLogger(name)
    if name not in self.__logger_names:
      self.__logger_names.add(name)
      self.configure_logger(logger)
    return logger

  def configure(
    self,
    *,
    terminal: bool,
    level: int
  ):
    self.level = level
    self.terminal = terminal
    for log_name in self.__logger_names:
      logger = logging.getLogger(log_name)
      self.configure_logger(logger)
      
      

class TimeLogger:
  start_time: float
  title: str
  logger: logging.Logger
  report_start: bool
  def __init__(self, log_name: str, title: str, *, report_start: bool = False) -> None:
    self.logger = RegisteredLogger().provision(log_name)
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