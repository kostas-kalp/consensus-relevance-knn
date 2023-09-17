import logging
logger = logging.getLogger(__package__)

import os
from pathlib import Path

def check_filepath(filepath, force=True, dir=False, make=False, warn=True):
    """
    Check that the argument filepath file-system object exists
    Args:
        filepath: (str or Path) to resolve to a file-system file or directory
        force: (bool) raise an exception if True
        dir: (bool) if True the filepath should be a directory
        make: (bool) if True, create the directory filepath
        warn: (bool) if True, log a warning message

    Returns:
        a Path object for the filepath or None if filepath does not resolve to a file-system file or directory
    """
    if not isinstance(filepath, (str, Path)):
        if force:
            raise ValueError(f'argument filepath should be str or Path')
        elif warn:
            logger.warning(f'argument filepath should be str or Path')
        return None
    filepath = Path(filepath)
    msg = None
    # filepath = Path(filepath) if isinstance(filepath, str) or not isinstance(filepath, Path) else filepath
    if not dir:
        if not (filepath.exists() and filepath.is_file()):
            msg = f'file {filepath} does not exist'
    else:
        if not (filepath.exists() and filepath.is_dir()):
            msg = f'directory {filepath} does not exist'
            if make:
                os.makedirs(filepath, exist_ok=True)
                if warn:
                    logger.info(f'creating directory {filepath}')
                return check_filepath(filepath, force=force, make=False, dir=True, warn=warn)
    if msg is not None:
        if force:
            raise ValueError(msg)
        if warn:
            logger.warning(msg)
        filepath = None
    return filepath
