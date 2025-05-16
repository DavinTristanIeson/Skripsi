from dataclasses import dataclass
import logging
from typing import Optional

from modules.baseclass import Singleton
from modules.logger.handlers import LoggingBehaviorManager


@dataclass
class LoggerDisabledContext:
  loggers: list[logging.Logger]
  original_behavior: LoggingBehaviorManager

  def __enter__(self):
    new_behavior = LoggingBehaviorManager(terminal=False, level=logging.WARNING, file=None)
    for logger in self.loggers:
      new_behavior.apply(logger)

  def __exit__(self, *args):
    for logger in self.loggers:
      self.original_behavior.apply(logger)

class ProvisionedLogger(metaclass=Singleton):
  __logger_names: set[str]
  __logging_behavior: LoggingBehaviorManager

  def __init__(self):
    super().__init__()
    self.__logger_names = set()
    self.__logging_behavior = LoggingBehaviorManager(
      level=logging.INFO,
      terminal=False,
      file=None
    )

  def provision(self, name: str)->logging.Logger:
    logger = logging.getLogger(name)
    if name not in self.__logger_names:
      self.__logger_names.add(name)
      self.__logging_behavior.apply(logger)
    return logger
  
  def disable(self, names: list[str]):
    return LoggerDisabledContext(
      loggers=list(map(self.provision, names)),
      original_behavior=self.__logging_behavior,
    )

  def configure(
    self,
    *,
    terminal: bool,
    level: int,
    file: Optional[str],
  ):
    self.__logging_behavior.level = level
    self.__logging_behavior.terminal = terminal
    self.__logging_behavior.file = file
    for log_name in self.__logger_names:
      logger = logging.getLogger(log_name)
      self.__logging_behavior.apply(logger)
      

__all__ = [
  "ProvisionedLogger"
]