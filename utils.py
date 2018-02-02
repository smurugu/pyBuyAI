import logging
import os
import datetime as dt

def setup_logging(concat_single_log=True):
    """
    Set up log details: output files, logging text format, error level etc
    :param concat_single_log:
    :return:
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('log')
    log_folder = os.path.join(os.path.dirname(__file__), 'log_files')

    if concat_single_log:
        log_file = os.path.join(log_folder, 'log.txt')
    else:
        log_file = os.path.join(log_folder, 'log_{0}.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')))

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
