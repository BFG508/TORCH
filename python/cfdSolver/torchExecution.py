"""Bounded, deterministic CPU execution for independent flux batches."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def validate_workers(workers):
    if (
        not isinstance(workers, int)
        or isinstance(workers, bool)
        or not 1 <= workers <= 64
    ):
        raise ValueError("Workers must be an integer from 1 to 64")
    return workers


class Execution:
    """Lazy thread pool; ordered results and at most workers live tasks.
    """

    def __init__(self, workers=1):
        self.pool = None
        self.workers = validate_workers(workers)

    def map(self, function, items):
        if self.workers == 1:
            yield from map(function, items)
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="torch-flux"
            )
        iterator = iter(items)
        pending = deque()
        sentinel = object()
        try:
            for _ in range(self.workers):
                item = next(iterator, sentinel)
                if item is sentinel:
                    break
                pending.append(self.pool.submit(function, item))
            while pending:
                yield pending.popleft().result()
                item = next(iterator, sentinel)
                if item is not sentinel:
                    pending.append(self.pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()
            # A failed step must not leave tasks using its
            # arrays behind.
            for future in pending:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __deepcopy__(self, memo):
        return Execution(self.workers)

    def __del__(self):
        self.close()
