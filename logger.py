import logging
import os
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(self, config: dict):
        os.makedirs(os.path.dirname(config["file"]), exist_ok=True)
        self._log = logging.getLogger("smoking")
        self._log.setLevel(getattr(logging, config["level"]))
        handler = RotatingFileHandler(
            config["file"],
            maxBytes=config["max_size_mb"] * 1024 * 1024,
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        self._log.addHandler(handler)

    def info(self, msg):
        self._log.info(msg)

    def alert(self, *args):
        self._log.warning(str(args))