import os
import logging


def set_log_file(file_path, debug=True):
    if not os.path.exists(file_path):
        open(file_path, 'w').close()

    logging.basicConfig(
        filename=file_path,
        format='%(levelname)s %(asctime)s: %(name)s - %(message)s ',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG if debug else logging.INFO
    )
