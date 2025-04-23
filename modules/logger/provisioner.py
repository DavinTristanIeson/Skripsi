import logging

from modules.baseclass import Singleton
from modules.logger.handlers import LoggingBehaviorManager


class LoggerDisabledContext:
  loggers: list[logging.Logger]
  __original_behaviors: list[LoggingBehaviorManager]
  def __init__(self, loggers: list[logging.Logger]):
    self.loggers = loggers
    self.__original_behaviors = list(map(LoggingBehaviorManager.infer, loggers))

  def __enter__(self):
    new_behavior = LoggingBehaviorManager(terminal=False, level=logging.WARNING)
    for logger in self.loggers:
      new_behavior.apply(logger)

  def __exit__(self, *args):
    for logger, behavior in zip(self.loggers, self.__original_behaviors):
      behavior.apply(logger)

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
  
  def disable(self, names: list[str]):
    return LoggerDisabledContext(list(map(self.provision, names)))

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