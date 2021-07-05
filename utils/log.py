"""
Utilities for logging.
"""

import os
from functools import partial
from .utils import makedirs, wrap_args


def _print_log(save_dir, log_filename, print_str):
    """
    Print to stdout and flush output to file.
    """
    print(print_str)
    with open(os.path.join(save_dir, log_filename), "a") as log_file:
        log_file.write(str(print_str) + "\n")


@wrap_args
def get_logger(save_dir, log_filename, **_):
    """
    Set up logging and return closure for logging.
    """
    makedirs(save_dir)
    return partial(_print_log, save_dir, log_filename)
