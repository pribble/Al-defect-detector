import os
import logging

def setup_log(name=__name__):
  if os.path.exists('/var/logs') == False:
    os.makedirs('/var/logs', exist_ok=True)
  if os.path.exists('/var/logs/api_server.log'):
    size = os.stat('/var/logs/api_server.log').st_size
    if size >= 10*1024*1024:
      # 打开文件
      with open("/var/logs/api_server.log", "r+") as f:
        f.truncate(int(size / 2))
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter('[%(asctime)s %(filename)s:%(funcName)s:%(lineno)d %(levelname)s]->%(message)s')

  filename = "/var/logs/api_server.log"
  handler = logging.FileHandler(filename)
  handler.setLevel(logging.INFO)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  return logger
