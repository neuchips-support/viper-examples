import logging
import os
from datetime import datetime
import llm_rag_config as config

# Define log directory and file size limit
LOG_DIR = "data/logs"
LOG_FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10MB

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)


def get_log_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"user_activity_{timestamp}.txt")


def get_current_log_file():
    log_files = sorted(
        [
            f
            for f in os.listdir(LOG_DIR)
            if f.startswith("user_activity_") and f.endswith(".txt")
        ],
        key=lambda f: os.path.getmtime(
            os.path.join(LOG_DIR, f)
        ),  # Sort by last modified time
        reverse=True,
    )

    if log_files:
        latest_log = os.path.join(LOG_DIR, log_files[0])
        if os.path.getsize(latest_log) < LOG_FILE_SIZE_LIMIT:
            return latest_log

    return get_log_filename()


def create_file_handler(log_file):
    try:
        # remove all the handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        file_handler = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    except (OSError, IOError) as e:
        print(f"Error creating file handler: {e}")
        return None


def log_action(action):
    global LOG_FILE

    if config.user_activity_logging:
        # Check if log file exceeds size limit
        if (
            os.path.exists(LOG_FILE)
            and os.path.getsize(LOG_FILE) >= LOG_FILE_SIZE_LIMIT
        ):
            LOG_FILE = get_log_filename()
            create_file_handler(LOG_FILE)

        logger.info(action)


# Initialize logging
LOG_FILE = get_current_log_file()
LOGGER_NAME = "UserLogger"

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    logger.setLevel(logging.INFO)
    create_file_handler(LOG_FILE)
