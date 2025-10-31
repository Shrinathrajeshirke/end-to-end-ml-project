import logging
import os 
from datetime import datetime

### define logfile name
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

### path of log file
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

### make directory for log path
os.makedirs(logs_path, exist_ok=True)

### log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

### logging basic config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]-%(lineno)d-%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO,
)

#if __name__=="__main__":
#    logging.info("logging has started")
