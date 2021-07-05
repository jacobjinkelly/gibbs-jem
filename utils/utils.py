"""
Miscellaneous utilities.
"""

import os
import json


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def log_config(args):
    """
    Log job configuration.
    """
    with open("{}/args.txt".format(args.save_dir), 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)


def wrap_args(f):
    return lambda parse_args, *args, **kwargs: f(*args, **kwargs, **vars(parse_args))
