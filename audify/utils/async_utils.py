"""
Async utilities for concurrent API operations.

This module provides helpers for running async operations with concurrency limits
and integrating async code with synchronous contexts.
"""

import asyncio
from typing import Any, Coroutine, List, TypeVar, Union

T = TypeVar("T")


class AsyncBatcher:
    """Manages concurrent execution of async operations with limits.

    This class provides a way to run multiple async operations concurrently
    while limiting the number of simultaneous operations to prevent
    overwhelming APIs with too many requests.

    Attributes:
        max_concurrent: Maximum number of operations to run simultaneously.

    Example:
        >>> batcher = AsyncBatcher(max_concurrent=5)
        >>> results = await batcher.run_batch([
        ...     fetch_data(url1),
        ...     fetch_data(url2),
        ...     fetch_data(url3),
        ... ])
    """

    def __init__(self, max_concurrent: int = 10):
        """Initialize the batcher with a concurrency limit.

        Args:
            max_concurrent: Maximum number of concurrent operations.
                Defaults to 10.
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_batch(
        self, coros: List[Coroutine[Any, Any, T]]
    ) -> List[Union[T, BaseException]]:
        """Run a batch of coroutines with concurrency limiting.

        Args:
            coros: List of coroutines to execute.

        Returns:
            List of results in the same order as the input coroutines.
            Failed operations return the exception instead of raising.
        """

        async def limited(coro: Coroutine[Any, Any, T]) -> T:
            async with self.semaphore:
                return await coro

        return await asyncio.gather(
            *[limited(c) for c in coros], return_exceptions=True
        )

    async def run_batch_ordered(
        self,
        coros: List[Coroutine[Any, Any, T]],
        return_exceptions: bool = False,
    ) -> List[Union[T, BaseException]]:
        """Run a batch of coroutines preserving order.

        Args:
            coros: List of coroutines to execute.
            return_exceptions: If True, exceptions are returned as results.
                If False (default), first exception is raised.

        Returns:
            List of results in the same order as input.
        """

        async def limited(coro: Coroutine[Any, Any, T]) -> T:
            async with self.semaphore:
                return await coro

        return await asyncio.gather(
            *[limited(c) for c in coros], return_exceptions=return_exceptions
        )


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a synchronous context.

    This helper allows calling async functions from sync code. It handles
    the case where an event loop may or may not already be running.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.

    Example:
        >>> async def fetch_data():
        ...     return "data"
        >>> result = run_async(fetch_data())
        >>> print(result)
        'data'
    """
    try:
        # Check if there's already a running event loop
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, we can use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop - we need to handle this carefully
    # This can happen when called from within an async context
    # Use nest_asyncio pattern or run in thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


async def gather_with_limit(
    coros: List[Coroutine[Any, Any, T]],
    limit: int = 10,
    return_exceptions: bool = True,
) -> List[Union[T, BaseException]]:
    """Convenience function to gather coroutines with a concurrency limit.

    Args:
        coros: List of coroutines to execute.
        limit: Maximum concurrent operations.
        return_exceptions: Whether to return exceptions or raise them.

    Returns:
        List of results or exceptions.
    """
    batcher = AsyncBatcher(max_concurrent=limit)
    if return_exceptions:
        return await batcher.run_batch(coros)
    return await batcher.run_batch_ordered(coros, return_exceptions=False)
