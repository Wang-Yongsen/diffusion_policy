"""Single-key terminal input with state restoration on exit."""

import os
import select
import sys
import termios
import tty


class MotionTerminal:
    """回位期间读取单键；启动前丢掉已经排队的回车。"""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream

    def __enter__(self):
        if not self.stream.isatty():
            raise ValueError("需要交互终端；请直接运行命令，不要重定向 stdin")
        self.fd = self.stream.fileno()
        self.original = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)
        except BaseException:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
            raise
        return self

    def read(self, timeout):
        if not select.select([self.fd], [], [], max(0.0, timeout))[0]:
            return None
        data = os.read(self.fd, 32)
        return data.decode("utf-8", errors="ignore") if data else ""

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
