import time
import asyncio

def timeit_listener(fn):
    if asyncio.iscoroutinefunction(fn):
        async def wrapper(self, state):
            t0 = time.perf_counter()
            out = await fn(self, state)
            elapsed = time.perf_counter() - t0
            print(f"[TIMING] {fn.__name__:20s} -> {elapsed*1000:7.1f}ms")
            return out
    else:
        def wrapper(self, state):
            t0 = time.perf_counter()
            out = fn(self, state)
            elapsed = time.perf_counter() - t0
            print(f"[TIMING] {fn.__name__:20s} -> {elapsed*1000:7.1f}ms")
            return out
    return wrapper

