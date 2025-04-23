from dataclasses import dataclass
import logging
import sys

TERMINAL_STREAM_HANDLER = logging.StreamHandler(sys.stdout)
TERMINAL_STREAM_HANDLER.setFormatter(logging.Formatter('\033[38;5;247m%(asctime)s %(levelname)s\033[0m \033[1m[%(name)s]\033[0m: %(message)s'))
@dataclass
class LoggingBehaviorManager:
  level: int
  terminal: bool
  def apply(self, logger: logging.Logger):
    logger.setLevel(self.level)
    if self.terminal:
      logger.addHandler(TERMINAL_STREAM_HANDLER)
    else:
      logger.removeHandler(TERMINAL_STREAM_HANDLER)

  @staticmethod
  def infer(logger: logging.Logger):
    return LoggingBehaviorManager(
      level=logger.level,
      terminal=TERMINAL_STREAM_HANDLER in logger.handlers
    )

__all__ = [
  "LoggingBehaviorManager"
]