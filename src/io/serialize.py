import sys
from functools import wraps
from pathlib import Path

from joblib import load, dump


def cached(fname):
    """Allow the result of a function to be serialized and add a key word arg to allow invalidating
    the serialized object. To invalidate the serialized result pass invalidate=True to the decorated
    function.

    If the file containing the serialized result is not found, the decorated function will be ran as
    normal and the result is serialized.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, invalidate=False, **kwargs):
            cachepath = Path(f"./serialized/{fname}.pkl")

            if invalidate is False and cachepath.exists():
                try:
                    return load(cachepath)
                except Exception as e:
                    print(f"{e}\n\nCould not deserialize {cachepath}. Recomputing...",
                          file=sys.stderr)
                    if cachepath.exists():
                        cachepath.unlink()  # Delete the file.

            res = f(*args, **kwargs)
            dump(res, cachepath)
            return res

        return wrapper

    return decorator
