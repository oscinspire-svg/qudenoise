"""Tensor helpers, SVD truncation, and logging setup.

All array work goes through :mod:`qudenoise.backend`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from .backend import DTYPE, get_backend, to_backend, to_numpy

logger = logging.getLogger("qudenoise")

#: Singular values (of the *normalized* state) below this are numerical zeros.
DEFAULT_SVD_CUTOFF = 1e-14


def setup_logging(level: int | str = logging.INFO) -> None:
    """Attach a simple stream handler to the ``qudenoise`` logger (idempotent)."""
    if not any(getattr(h, "_qudenoise", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._qudenoise = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)


# --------------------------------------------------------------------------
# Linear algebra
# --------------------------------------------------------------------------
def safe_svd(matrix: Any) -> Tuple[Any, Any, Any]:
    """Thin SVD ``(U, S, Vh)`` with a robust fallback.

    LAPACK's default ``gesdd`` driver occasionally fails to converge on
    ill-conditioned matrices; in that case retry on the host with the slower
    but more robust ``gesvd`` driver from SciPy and move the result back.
    """
    xp = get_backend()
    try:
        return xp.linalg.svd(matrix, full_matrices=False)
    except Exception as exc:  # LinAlgError (numpy) / backend-specific errors
        logger.warning("SVD failed (%s); retrying with scipy gesvd fallback", exc)
        from scipy.linalg import svd as scipy_svd

        u, s, vh = scipy_svd(to_numpy(matrix), full_matrices=False, lapack_driver="gesvd")
        return to_backend(u, DTYPE), to_backend(s), to_backend(vh, DTYPE)


def svd_truncate(
    matrix: Any,
    chi_max: Optional[int] = None,
    threshold: Optional[float] = None,
    cutoff: float = DEFAULT_SVD_CUTOFF,
) -> Tuple[Any, Any, Any, float]:
    """SVD of ``matrix`` followed by truncation.

    Truncation rules (applied together):

    * ``chi_max``   -- hard cap on the number of kept singular values.
    * ``threshold`` -- discard singular values ``s_i < threshold``, where the
      singular values are first normalized so that ``sum(s_i**2) == 1``.
    * ``cutoff``    -- same as ``threshold`` but always on; removes numerical
      zeros so bond dimensions do not grow spuriously.

    At least one singular value is always kept. The kept singular values are
    rescaled so the Frobenius norm of the matrix is preserved.

    Returns ``(U, S, Vh, discarded_weight)`` where ``discarded_weight`` is
    ``sum(s_discarded**2) / sum(s**2)`` -- the probability weight thrown away.
    """
    u, s, vh = safe_svd(matrix)
    s_host = to_numpy(s)
    weights = s_host.real**2
    total = float(weights.sum())
    if not total > 0.0:
        raise ValueError("Cannot truncate the SVD of an all-zero (or NaN) matrix.")
    limit = max(cutoff, threshold or 0.0)
    normalized = s_host / total**0.5
    keep = int((normalized >= limit).sum())  # s is sorted descending
    keep = max(1, keep)
    if chi_max is not None:
        keep = max(1, min(keep, int(chi_max)))
    discarded = float(weights[keep:].sum()) / total
    if keep < len(s_host):
        kept_weight = float(weights[:keep].sum())
        s = s[:keep] * (total / kept_weight) ** 0.5
        u, vh = u[:, :keep], vh[:keep, :]
    return u, s, vh, discarded


def reorder_two_qubit_matrix(matrix: Any) -> Any:
    """Re-express a two-qubit operator acting on (q0, q1) as acting on (q1, q0)."""
    return matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)


def is_unitary(matrix: Any, tol: float = 1e-10) -> bool:
    xp = get_backend()
    d = matrix.shape[0]
    return bool(abs(matrix.conj().T @ matrix - xp.eye(d)).max() < tol)


def dagger(matrix: Any) -> Any:
    return matrix.conj().T


def index_to_bitstring(index: int, n: int) -> str:
    """Big-endian: qubit 0 is the leftmost (most significant) bit."""
    return format(index, f"0{n}b")
