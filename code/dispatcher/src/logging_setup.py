"""
Sets up a logger that writes live, unbuffered progress to both a log file
and the console — so a long unattended run can be checked on mid-flight.
"""

import os
import sys
import logging
from datetime import datetime

import constants as c


def setup_logging():
    os.makedirs(c.LOGS_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(c.LOGS_DIR, f"nsga2_ins_loss_{run_id}.log")

    logger = logging.getLogger("nsga2_ins_loss")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_file}")
    return logger
