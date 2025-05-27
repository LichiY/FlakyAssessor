import logging
import sys
logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout)])
def debug(msg):
    logging.debug(msg)
def info(msg):
    logging.info(msg)
def warning(msg):
    logging.warning(msg)
def error(msg):
    logging.error(msg)
def fatal(msg):
    logging.critical(msg)
class Logger(object):
    def __init__(self, config):
        """
        Args:
            config: An object (e.g., Configure instance) with `log.filename` and `log.level` attributes.
        """
        super(Logger, self).__init__()
        log_level_str = config.get('log', {}).get('level', 'info')
        log_filename = config.get('log', {}).get('filename', None)
        if log_level_str not in logging_level:
            print(f"WARNING: Invalid log level '{log_level_str}'. Defaulting to 'info'.")
            log_level_str = 'info'
        level = logging_level[log_level_str]
        log_format = '%(asctime)s - %(levelname)s : %(message)s'
        date_format = '%Y/%m/%d %H:%M:%S'
        logging.getLogger('').handlers = []
        if log_filename:
            import os
            log_dir = os.path.dirname(log_filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_filename, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            logging.getLogger('').addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger('').addHandler(stream_handler)
        logging.getLogger('').setLevel(level)
        info("Logger initialized.")
        info(f"Logging level set to: {log_level_str.upper()}")
        if log_filename:
            info(f"Logging to file: {log_filename}")
        else:
            info("Logging to console only.")
