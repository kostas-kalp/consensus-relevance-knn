import functools
import logging

class LogLevel2(object):
    """
    set the log level for the logger to the given level temporarily while the decorated function executes
    restore the logger's level to what it was before
    @LogLevel(logger=logging.getLogger(__package__), level=logging.DEBUG)
    def function_to_debug(*args, **kwargs);
      ....

    """
    def __init__(self, logger=None, level=None):
        self.level = level
        self.logger = logger
        pass

    def __call__(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            try:
                old_level = None
                if self.level is not None and self.logger is not None:
                    old_level = self.logger.getEffectiveLevel()
                    self.logger.setLevel(self.level)
                    result = fn(*args, **kwargs)
                    self.logger.setLevel(old_level)
                else:
                    result = fn(*args, **kwargs)
                return result
            except Exception as ex:
                if old_level is not None and self.logger is not None:
                    self.logger.setLevel(old_level)
                self.logger.exception(ex, exc_info=True)
                raise ex
            return result
        return decorated


def LogLevel(logger=None, level=None):
    # Same as the LogLevel2 class but as a function
    def decorator(fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            try:
                old_level = None
                if level is not None and logger is not None:
                    old_level = logger.getEffectiveLevel()
                    logger.setLevel(level)
                    result = fn(*args, **kwargs)
                    logger.setLevel(old_level)
                else:
                    result = fn(*args, **kwargs)
                return result
            except Exception as ex:
                if old_level is not None and logger is not None:
                    logger.setLevel(old_level)
                logger.exception(ex, exc_info=True)
                raise ex
            return result
        return decorated
    return decorator
