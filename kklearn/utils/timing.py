import logging
logger = logging.getLogger(__package__)

from functools import wraps
from collections import defaultdict

from datetime import datetime
import time


def timestamp_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def timed(method=False):
    # outer decorator with arguments
    #
    # If method is True then it appends the measured time to the list for the method maintained in the exec_times_ dict
    # at the instance (the dict is keyed on the method's name)
    # If method is False, just print the measured time
    #
    def timed_no_arguments(func):
        # inner decorator with no arguments
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            t = round(end - start, 6)
            if method and args[0] is not None:
                obj = args[0]
                name = func.__name__
                if not hasattr(obj, "exec_times_"):
                    obj.exec_times_ = defaultdict(list)
                if hasattr(obj, "exec_times_"):
                    obj.exec_times_[name].append(t)
            else:
                print(f"method {name}() ran in {t} sec")
            return result
        return wrapper

    return timed_no_arguments



class ScopeTimer():
    """Context manager to measure how much time was spent in the target scope."""
    #  https://stackoverflow.com/questions/54076972/returning-value-when-exiting-python-context-manager
    #  timer = time_this_scope()
    #  with timer:
    #       x = 1
    # print(f'timer dt={timer.dt} {timer.unit}')

    def _look(self):
        # v = np.asarray([time.process_time_ns(), time.perf_counter_ns()])/self.scale
        v = time.process_time_ns()/self.scale
        return v

    def __init__(self, verbose=False, name='cputime', scale=1e6, unit='ms'):
        self.t0 = None
        self.dt = None
        self.verbose = verbose
        self.name = name
        self.scale = scale
        self.unit=unit

    def __enter__(self):
        self.t0 = self._look()

    def __exit__(self, type=None, value=None, traceback=None):
        self.dt = self._look() - self.t0
        if self.verbose is True:
            print(f"scope took {self.name}={self.dt} {self.unit}")

    def __repr__(self):
        return f'dt={timer.dt} {timer.unit}'