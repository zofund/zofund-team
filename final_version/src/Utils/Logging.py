import datetime
import logging.config
import logging
import os
from src.LocalEnv import LOG_DIR


class MyLog:
    # 定义日志输出格式
    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    # 定义日志文件的路径
    LOG_PATH = os.path.join(LOG_DIR, '{:%Y-%m-%d}.log'.format(datetime.datetime.today()))

    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setFormatter(MyLog.fmt)
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(MyLog.fmt)
        sh.setLevel(logging.DEBUG)
        fh = logging.FileHandler(MyLog.LOG_PATH)
        fh.setFormatter(MyLog.fmt)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


LOGGER = MyLog().logger

if __name__ == '__main__':
    LOGGER.info('test')

