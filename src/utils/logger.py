import logging
from pathlib import Path


def get_logger(out_dir: Path, logname: str) -> logging.Logger:
    fh = logging.FileHandler(out_dir.joinpath(logname), "w")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    log = logging.getLogger("root")
    log.setLevel(logging.INFO)
    log.handlers = []  # No duplicated handlers
    log.propagate = False  # workaround for duplicated logs in ipython
    log.addHandler(ch)
    log.addHandler(fh)
    return log
