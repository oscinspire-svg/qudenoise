"""Array-backend abstraction layer (NumPy <-> CuPy).

This is the *only* library module that imports NumPy. Every other module
obtains the active array library through :func:`get_backend` so that all
tensor code is device-agnostic::

    from qudenoise.backend import get_backend as xp_
    xp = xp_()
    xp.einsum("ab,bc->ac", A, B)

CuPy is strictly optional. It is imported lazily, only when GPU support is
probed (``device="auto"``) or explicitly requested, so a NumPy-only
installation imports and runs with no errors or warnings about CuPy.

The active backend is process-global state (like matplotlib's backend).
Use :func:`use_backend` for scoped switches.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterator, Optional

import numpy as _np

logger = logging.getLogger("qudenoise")

#: dtype used for all state tensors and gates.
DTYPE = "complex128"

_cp: Any = None  # the cupy module once successfully loaded
_cupy_probed = False
_cupy_error: Optional[str] = None

_current: Any = _np
_current_name = "numpy"

_ALIASES = {
    "numpy": "numpy",
    "np": "numpy",
    "cpu": "numpy",
    "cupy": "cupy",
    "cuda": "cupy",
    "gpu": "cupy",
}


# --------------------------------------------------------------------------
# CuPy detection
# --------------------------------------------------------------------------
def _load_cupy() -> Any:
    """Import CuPy lazily. Returns the module or ``None``; never raises."""
    global _cp, _cupy_probed, _cupy_error
    if _cupy_probed:
        return _cp
    _cupy_probed = True
    try:
        import cupy  # type: ignore

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy is installed but no CUDA device was found")
        _cp = cupy
    except Exception as exc:  # ImportError, CUDA runtime errors, ...
        _cp = None
        _cupy_error = f"{type(exc).__name__}: {exc}"
    return _cp


def gpu_available() -> bool:
    """True if CuPy is importable *and* a CUDA device is visible."""
    return _load_cupy() is not None


def _is_cupy_array(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "cupy"


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
def _normalize(name: str) -> str:
    try:
        return _ALIASES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown backend: {name!r} (expected 'numpy' or 'cupy')"
        ) from None


def set_backend(name: str) -> Any:
    """Activate ``'numpy'`` or ``'cupy'`` (aliases: cpu / cuda / gpu).

    Raises ``RuntimeError`` if CuPy is requested but unavailable.
    Returns the activated array module.
    """
    global _current, _current_name
    canon = _normalize(name)
    if canon == "cupy":
        cp = _load_cupy()
        if cp is None:
            raise RuntimeError(
                "CuPy backend requested but CuPy is not installed or no GPU "
                f"was detected ({_cupy_error}). Install a CuPy wheel matching "
                "your CUDA toolkit, e.g. `pip install cupy-cuda12x`."
            )
        _current, _current_name = cp, "cupy"
    else:
        _current, _current_name = _np, "numpy"
    if _debug_enabled():
        validate_svd()
    return _current


def get_backend() -> Any:
    """Return the active array module (``numpy`` or ``cupy``)."""
    return _current


def backend_name() -> str:
    """``'numpy'`` or ``'cupy'``."""
    return _current_name


def is_gpu() -> bool:
    return _current_name == "cupy"


def resolve_device(device: str = "auto") -> str:
    """Map a user-facing device string to a backend name.

    ``"auto"`` -> CuPy when available, else NumPy (never fails).
    ``"cpu"``/``"numpy"`` -> NumPy.
    ``"cuda"``/``"gpu"``/``"cupy"`` -> CuPy, raising if unavailable.
    """
    if str(device).lower() == "auto":
        return "cupy" if gpu_available() else "numpy"
    canon = _normalize(device)
    if canon == "cupy" and not gpu_available():
        raise RuntimeError(
            f"device={device!r} requested but CuPy/GPU is unavailable "
            f"({_cupy_error}). Use device='auto' to fall back to CPU."
        )
    return canon


def configure(device: str = "auto", *, log: bool = True) -> str:
    """Resolve ``device``, activate it, and (optionally) log the choice.

    Returns the active backend name.
    """
    name = resolve_device(device)
    set_backend(name)
    if log:
        if name == "cupy":
            logger.info("QuDenoise: running on GPU (CuPy)")
        elif str(device).lower() == "auto":
            logger.info("QuDenoise: no GPU/CuPy found, running on CPU (NumPy)")
        else:
            logger.info("QuDenoise: running on CPU (NumPy)")
    return name


@contextlib.contextmanager
def use_backend(name: str) -> Iterator[Any]:
    """Temporarily switch backend inside a ``with`` block."""
    prev = _current_name
    xp = set_backend(name)
    try:
        yield xp
    finally:
        set_backend(prev)


# --------------------------------------------------------------------------
# Host <-> device movement
# --------------------------------------------------------------------------
def to_backend(array: Any, dtype: Optional[str] = None) -> Any:
    """Move / convert ``array`` onto the active backend."""
    if _current_name == "cupy":
        out = _current.asarray(array)  # numpy->cupy copies, cupy->cupy no-op
    else:
        out = _np.asarray(array.get() if _is_cupy_array(array) else array)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def to_numpy(array: Any) -> Any:
    """Return a host ``numpy.ndarray`` regardless of where ``array`` lives."""
    if _is_cupy_array(array):
        return array.get()
    return _np.asarray(array)


# --------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------
def get_rng(seed: Optional[int] = None) -> Any:
    """Backend-appropriate ``Generator`` (``default_rng``), explicitly seeded.

    There is no global RNG state anywhere in the library: generators are
    created here and passed explicitly to whoever needs randomness.
    """
    return _current.random.default_rng(seed)


def spawn_seeds(seed: Optional[int], n: int) -> list[int]:
    """Derive ``n`` statistically independent integer seeds from ``seed``.

    Trajectory ``i`` always receives the same seed for a given master seed,
    independent of how trajectories are scheduled across workers.
    """
    ss = _np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype="uint64")[0]) for child in ss.spawn(n)]


def draw_uniform(rng: Any) -> float:
    """One uniform ``[0, 1)`` sample as a Python float, on either backend."""
    return float(rng.random())


# --------------------------------------------------------------------------
# SVD self-check
# --------------------------------------------------------------------------
def _debug_enabled() -> bool:
    return os.environ.get("QUDENOISE_DEBUG", "").lower() in ("1", "true", "yes", "on")


def validate_svd(tol: float = 1e-10, seed: int = 1234) -> None:
    """Check ``linalg.svd`` on the active backend across several shapes.

    A silently wrong SVD would corrupt every truncation step, so this runs
    automatically on backend switches when ``QUDENOISE_DEBUG=1`` and can be
    called explicitly at any time. Raises ``RuntimeError`` on failure.
    """
    xp = _current
    rng = _np.random.default_rng(seed)
    for shape in [(2, 2), (4, 4), (8, 4), (4, 8), (16, 16), (32, 8), (1, 4)]:
        host = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        m = to_backend(host, DTYPE)
        u, s, vh = xp.linalg.svd(m, full_matrices=False)
        rec = (u * s) @ vh
        err = float(abs(rec - m).max())
        orth = float(abs(u.conj().T @ u - xp.eye(u.shape[1])).max())
        unsorted = bool((s[:-1] < s[1:] - 1e-12).any())
        if err >= tol or orth >= tol or unsorted:
            raise RuntimeError(
                f"SVD self-check failed on backend {_current_name!r} for shape "
                f"{shape}: reconstruction error {err:.2e}, "
                f"orthogonality error {orth:.2e}, unsorted={unsorted}"
            )


if _debug_enabled():  # opt-in import-time check
    validate_svd()
