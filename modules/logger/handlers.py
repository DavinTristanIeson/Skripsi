from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Optional

from modules.baseclass import Singleton

DEFAULT_FORMATTER = logging.Formatter('\033[38;5;247m%(asctime)s %(levelname)s\033[0m \033[1m[%(name)s]\033[0m: %(message)s')
LOGGING_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')

TERMINAL_STREAM_HANDLER = logging.StreamHandler(sys.stdout)
TERMINAL_STREAM_HANDLER.setFormatter(DEFAULT_FORMATTER)


class RotatingFileHandlerProvisioner(metaclass=Singleton):
  __handlers: dict[str, RotatingFileHandler]
  def __init__(self):
    self.__handlers = {}

  def provision(self, file: str):
    if file not in self.__handlers:
      handler = RotatingFileHandler(
        filename=file,
        # 100 kB
        maxBytes=100 * 1000,
        backupCount=2
      )
      handler.setFormatter(LOGGING_FORMATTER)
      self.__handlers[file] = handler
    return self.__handlers[file]
    
  def remove_all_files(self, logger: logging.Logger):
    # pretty bad way to do this, but oh well
    for handler in self.__handlers.values():
      try:
        logger.removeHandler(handler)
      except ValueError:
        continue
  
  def remove_file_with_prefix(self, logger: logging.Logger):
    # pretty bad way to do this, but oh well
    for handler in self.__handlers.values():
      try:
        logger.removeHandler(handler)
      except ValueError:
        continue


@dataclass
class LoggingBehaviorManager:
  level: int
  terminal: bool
  file: Optional[str]

  def apply(self, logger: logging.Logger):
    logger.setLevel(self.level)
    if self.terminal:
      logger.addHandler(TERMINAL_STREAM_HANDLER)
    else:
      logger.removeHandler(TERMINAL_STREAM_HANDLER)
    if self.file is not None:
      handler = RotatingFileHandlerProvisioner().provision(self.file)
      logger.addHandler(handler)
    else:
      RotatingFileHandlerProvisioner().remove_all_files(logger)

__all__ = [
  "LoggingBehaviorManager"
]