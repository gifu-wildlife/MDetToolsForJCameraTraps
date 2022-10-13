import time
from datetime import datetime
from logging import Logger
from typing import Optional

import humanfriendly


class Timer(object):
    def __init__(
        self,
        verbose: bool = False,
        timer_tag: str = "Timer:",
        logger: Optional[Logger] = None,
    ) -> None:
        self.verbose = verbose
        self.logger = None if logger is None else logger
        self.timer_tag = timer_tag + " "

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        start_timestamp = datetime.fromtimestamp(self.start)
        end_timestamp = datetime.fromtimestamp(self.end)
        if self.verbose:
            if self.logger is not None:
                self.logger.info(
                    f"{self.timer_tag}Start Time: {start_timestamp.strftime('%Y/%m/%d %H:%M:%S')}"
                )
                self.logger.info(
                    f"{self.timer_tag}End Time: {end_timestamp.strftime('%Y/%m/%d %H:%M:%S')}"
                )
                # self.logger.info(f"{self.timer_tag}Elapsed Time: {self.msecs} ms ({self.msecs // 1000} sec)")
                self.logger.info(
                    f"{self.timer_tag}Elapsed Time: {humanfriendly.format_timespan(self.secs)}"
                )
            else:
                print(f"elapsed time: {self.msecs} ms")
