import logging
logger = logging.getLogger(__package__)

import hashlib

def digest(obj, length=4, consistent=True):
    """
    Compute a hash digest of an item as a hex string
    Args:
        obj: an item whose string representation will be hashed
        length: (int) the length of the desired hex hash value
        consistent: (bool) if True repeated calls on the same input give same hash value
                 (in same or different processes)

    Returns:
        a hex string of the hash of its argument
    """
    fn1 = lambda u: hashlib.sha256(u).hexdigest().upper()
    fn2 = lambda u: hashlib.md5(u).hexdigest().upper()
    fn3 = lambda u: hashlib.shake_256(u).hexdigest(length).upper()
    fn = fn3
    if obj is not None:
        u = f'{obj}'.encode()
        if not consistent:
            u += f'{obj.__hash__()}'.encode()
        s = fn(u)
        return s[:min(length, len(s))]
    return None
