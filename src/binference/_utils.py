"""Small internal helpers: random-number handling, validation and progress."""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

RNGLike = int | np.random.Generator | np.random.SeedSequence | None

#: frames under this directory are skipped, so warnings point at the caller's line
PACKAGE_DIR = (os.path.dirname(os.path.abspath(__file__)) + os.sep,)


def warn(message: str, category: type[Warning]) -> None:
    """Issue a warning attributed to the first line outside the package."""
    warnings.warn(message, category, skip_file_prefixes=PACKAGE_DIR)


def frozen(a: Any) -> Any:
    """``a`` made read-only (a dense array, or the arrays of a sparse CSR matrix)."""
    for arr in (a.data, a.indices, a.indptr) if hasattr(a, "indptr") else (a,):
        arr.setflags(write=False)
    return a


class SetOnce:
    """Base of the immutable classes: each slot is set once, when the object is made."""

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).__slots__ and not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"{type(self).__name__} objects are immutable.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} objects are immutable.")


def check_positive_integers(value: Any, name: str, n: int = 1) -> tuple[int, ...]:
    """``value``, one for all ``n`` axes or one per axis, as ``n`` positive integers (an
    integral float is accepted)."""
    try:
        v = np.broadcast_to(np.asarray(value), (n,))
        if all(float(x).is_integer() and x >= 1 for x in v):
            return tuple(int(x) for x in v)
    except (TypeError, ValueError):
        pass
    per_axis = f", or one per axis ({n})" if n > 1 else ""
    raise ValueError(f"{name} must be a positive integer{per_axis}; got {value!r}.")


def as_generator(rng: RNGLike = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` following SPEC 7.

    Parameters
    ----------
    rng : None, int, SeedSequence or Generator
        ``None`` draws fresh entropy from the OS; an ``int`` or
        :class:`~numpy.random.SeedSequence` seeds a new generator; a
        :class:`~numpy.random.Generator` is used as is (and therefore advanced).
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or isinstance(rng, int | np.integer | np.random.SeedSequence):
        return np.random.default_rng(rng)
    if isinstance(rng, np.random.RandomState):
        raise TypeError(
            "Legacy numpy.random.RandomState objects are not supported; pass an int seed "
            "or a numpy.random.Generator (e.g. numpy.random.default_rng(seed))."
        )
    raise TypeError(f"Cannot interpret {type(rng).__name__} as a random number generator.")


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Independent child generators (one per chain)."""
    return list(rng.spawn(n))


def chunk_slices(n: int, chunk: int | None) -> Iterator[slice]:
    """Yield slices covering ``range(n)`` in blocks of at most ``chunk``."""
    if chunk is None or chunk <= 0 or chunk >= n:
        yield slice(0, n)
        return
    for start in range(0, n, chunk):
        yield slice(start, min(start + chunk, n))


def check_positive_weights(weights: Any, n: int) -> np.ndarray:
    """Validate per-object weights (default ones)."""
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        w = np.full(n, float(w))
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},); got {w.shape}.")
    return check_positive(w, "weights")


def check_positive(value: Any, name: str) -> np.ndarray:
    """``value`` as a float array, after checking that it is finite and strictly positive."""
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return a


def check_nonnegative(value: Any, name: str) -> np.ndarray:
    """``value`` as a float array, after checking that it is finite and non-negative."""
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    return a


def is_integer_array(x: np.ndarray, atol: float = 1e-12) -> bool:
    """True if every entry of ``x`` is (numerically) an integer."""
    return bool(np.all(np.abs(x - np.round(x)) <= atol))


def progress(
    iterable: Iterable, total: int | None = None, desc: str = "", enable: bool = False
) -> Iterable:
    """Wrap an iterable with a progress bar (tqdm if available) when ``enable``."""
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - depends on environment
        return _TextProgress(iterable, total=total, desc=desc)
    return tqdm(iterable, total=total, desc=desc)


class _TextProgress:  # pragma: no cover - cosmetic
    """Minimal text progress reporting to stderr (no dependencies)."""

    def __init__(self, iterable: Iterable, total: int | None, desc: str) -> None:
        self.iterable = iterable
        self.total = total
        self.desc = desc

    def __iter__(self) -> Iterator:
        step = max(1, (self.total or 100) // 20)
        for i, item in enumerate(self.iterable):
            if i % step == 0:
                frac = f"{i}/{self.total}" if self.total else str(i)
                sys.stderr.write(f"\r{self.desc} {frac}")
                sys.stderr.flush()
            yield item
        sys.stderr.write("\n")
