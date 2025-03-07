import gzip
import http.client as http_client
import logging
import os
import time
import traceback
from logging.handlers import RotatingFileHandler


class GZipRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        """
        Override doRollover to compress the log file after rotation.
        """
        super().doRollover()
        # Compress the rotated log file
        if self.backupCount > 0:
            log_dir = os.path.dirname(self.baseFilename)
            # RotatingFileHandler names backup files as 'basename.log.1', 'basename.log.2', etc.
            # We need to compress the most recent backup file
            sfn = f"{self.baseFilename}.1"  # Most recent backup file
            if os.path.exists(sfn):
                with open(sfn, "rb") as f_in:
                    with gzip.open(f"{sfn}.gz", "wb") as f_out:
                        f_out.writelines(f_in)
                os.remove(sfn)
                # Optionally, you can log the compression action
                logging.getLogger("storydiff_logger").info(f"Compressed and removed {sfn}")


def setup_logger():
    logger = logging.getLogger("storydiff_logger")
    logger.setLevel(logging.DEBUG)

    # Create log directory if it doesn't exist
    log_dir = os.path.expanduser("~/logs/storydiff")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File handler for logging to a file with rotation and compression
    log_file = os.path.join(log_dir, "output.log")
    file_handler = GZipRotatingFileHandler(
        log_file,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=5,  # Keep up to 5 backup files
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Set up detailed logging for requests and urllib3
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").addHandler(file_handler)
    logging.getLogger("urllib3").addHandler(stream_handler)

    # Enable logging for http.client (this will show request/response headers)
    http_client.HTTPConnection.debuglevel = 1
    logging.getLogger("http.client").setLevel(logging.DEBUG)
    logging.getLogger("http.client").addHandler(file_handler)
    logging.getLogger("http.client").addHandler(stream_handler)

    # Cleanup old logs
    cleanup_old_logs(log_dir)

    return logger


def log_request_and_response(r, *args, **kwargs):
    """Log the request and response details."""
    logger = logging.getLogger("storydiff_logger")

    try:
        # Log request details
        request = r.request
        logger.debug(f"Request: {request.method} {request.url}")
        logger.debug(f"Request Headers: {request.headers}")
        if request.body:
            logger.debug(f"Request Body: {request.body}")

        # Log response details
        logger.debug(f"Response Status: {r.status_code}")
        logger.debug(f"Response Headers: {r.headers}")
        logger.debug(f"Response Body: {r.text}")

    except Exception as e:
        # Log exception with method name and line number
        tb = traceback.format_exc()
        logger.error(f"Exception occurred: {e}")
        logger.error(f"Traceback: {tb}")


def cleanup_old_logs(log_dir, pattern="output.log.*", days=5):
    """
    Deletes log files in the specified directory that are older than a certain number of days.

    :param log_dir: The directory where log files are stored.
    :param pattern: The pattern to match log files (supports glob patterns).
    :param days: The age in days beyond which log files should be deleted.
    """
    import glob

    logger = logging.getLogger("storydiff_logger")
    cutoff_time = time.time() - days * 86400  # 86400 seconds in a day

    # Get a list of log files matching the pattern
    # Include both uncompressed and compressed log files
    patterns = [pattern, f"{pattern}.gz"]
    log_files = []
    for pat in patterns:
        log_files.extend(glob.glob(os.path.join(log_dir, pat)))

    for log_file in log_files:
        if os.path.isfile(log_file):
            file_mtime = os.path.getmtime(log_file)
            if file_mtime < cutoff_time:
                try:
                    os.remove(log_file)
                    logger.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    logger.error(f"Failed to delete log file {log_file}: {e}")


# Setup logger
logger = setup_logger()
