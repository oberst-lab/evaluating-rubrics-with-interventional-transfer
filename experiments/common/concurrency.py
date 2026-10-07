"""Running one example's steps concurrently."""

from concurrent.futures import ThreadPoolExecutor


def parallel_map(work: list, task, max_workers: int) -> list:
    """Run `task` over `work` on a thread pool and return the results in `work` order."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(task, work))
