import os
import logging


def setup_log(name, log_file):
    if not os.path.exists('/var/logs'):
        os.makedirs('/var/logs', exist_ok=True)

    log_path = f'/var/logs/{log_file}'
    if os.path.exists(log_path):
        size = os.stat(log_path).st_size
        if size >= 10 * 1024 * 1024:
            with open(log_path, "r+") as f:
                f.truncate(int(size / 2))

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s:%(funcName)s:%(lineno)d %(levelname)s]->%(message)s'
    )

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
