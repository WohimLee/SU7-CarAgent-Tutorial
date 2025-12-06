import logging
import os
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# 创建目录
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("data_gen")
logger.setLevel(logging.DEBUG)


formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s]: %(message)s"
)

# 文件 handler
file_handler = RotatingFileHandler(
    "logs/data_gen_errors.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8"
)

file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)


# 控制台 handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(console_handler)
