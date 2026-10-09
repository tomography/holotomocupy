# logger_config.py
import sys
import logging
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[35m",      # magenta
    "INFO": "\033[32m",       # green
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m", # bold red
}

class MPIRankFilter(logging.Filter):
    """Tag records with the rank and drop rank>0 DEBUG records."""
    def filter(self, record):
        record.rank = rank
        if record.levelno == logging.DEBUG and rank != 0:
            return False
        return True


def add_file_handler(path: str):
    """Tee logger output to a plain-text file (no ANSI colours). Idempotent per path."""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == path:
            return h
    fh = logging.FileHandler(path, mode='w')
    fh.addFilter(MPIRankFilter())
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [rank=%(rank)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return fh

class ColorMessageFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        record.msg_colored = f"{color}{record.getMessage()}{RESET}"
        return super().format(record)

handler = logging.StreamHandler(sys.stdout)
handler.addFilter(MPIRankFilter())
handler.setFormatter(ColorMessageFormatter(
    fmt="%(asctime)s [rank=%(rank)d] %(msg_colored)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logger = logging.getLogger("")
logger.setLevel(logging.WARNING)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


def set_log_level(level: str):
    """Set the root logger level from a string (DEBUG/INFO/WARNING/ERROR)."""
    logger.setLevel(level.upper())
