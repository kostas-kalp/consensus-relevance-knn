import logging
logger = logging.getLogger(__package__)

# from sklearn.utils.extmath import weighted_mode

import sklearn.utils.extmath

def weighted_mode(a, w, axis=0):
    try:
        z = sklearn.utils.extmath.weighted_mode(a, w, axis=axis)
    except Exception as e:
        logger.warning(e, exc_info=True)
    return z
