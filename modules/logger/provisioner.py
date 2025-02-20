import logging

from modules.baseclass import Singleton
from modules.logger.handlers import LoggingBehaviorManager


class ProvisionedLogger(metaclass=Singleton):
  __logger_names: set[str]
  __logging_handler: LoggingBehaviorManager

  def __init__(self):
    super().__init__()
    self.__logger_names = set()
    self.__logging_handler = LoggingBehaviorManager(
      level=logging.INFO,
      terminal=False
    )

  def provision(self, name: str)->logging.Logger:
    logger = logging.getLogger(name)
    if name not in self.__logger_names:
      self.__logger_names.add(name)
      self.__logging_handler.apply(logger)
    return logger

  def configure(
    self,
    *,
    terminal: bool,
    level: int
  ):
    self.__logging_handler.level = level
    self.__logging_handler.terminal = terminal
    for log_name in self.__logger_names:
      logger = logging.getLogger(log_name)
      self.__logging_handler.apply(logger)
      

__all__ = [
  "ProvisionedLogger"
]