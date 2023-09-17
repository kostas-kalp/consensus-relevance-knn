import logging
logger = logging.getLogger(__package__)

import os
import sys
import traceback
import gc
import psutil

import numpy as np

def memory_usage(verbose=False):
    # return the memory usage in MB
    p = psutil.Process(os.getpid())
    with p.oneshot():
        t = [p.name(), p.cpu_times(), p.memory_info()]
        mem = np.asarray([p.memory_info().rss, p.memory_info().vms])
    s = float(2 ** 20)
    mem = mem / s
    if verbose:
        logger.info('Garbage collection stats\n\t' + '\n\t'.join([f'gen{i} {t}' for i, t in enumerate(gc.get_stats())]))
    return mem.round(decimals=3)
