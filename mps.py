"""Core Matrix Product State class.

Representation
--------------
``MPS.tensors[i]`` has shape ``(D_left, 2, D_right)`` (complex128) with
``D_left = 1`` at the first site and ``D_right = 1`` at the last. Qubit 0 is the
leftmost site and the *most significant* bit of the dense statevector index
(big-endian).

Canonical form
--------------
The MPS always tracks an *orthogonality center* ``mps.center``: tensors to
its left are left-isometries, tensors to its right are right-isometries, and
the center tensor carries the whole norm. This makes local truncations
optimal and local norms exact. Unitary single-qubit gates preserve this
structure at any site; two-qubit gates move the center next to the bond they
act on first (see :meth:`MPS.apply_2q`).

Truncation bookkeeping
----------------------
Every truncating SVD records its discarded probability weight ``eps_k``. The
running totals are exposed as

* :attr:`MPS.truncation_error` -- ``sum_k eps_k``
* :attr:`MPS.fidelity_estimate` -- ``prod_k (1 - eps_k)`` (the usual estimate)
* :attr:`MPS.fidelity_lower_bound` -- ``max(0, 1 - (sum_k sqrt(eps_k))**2)``, a
  rigorous bound on the fidelity with the untruncated state.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

from .backend import DTYPE, configure, get_backend, to_backend
from .utils import (
    DEFAULT_SVD_CUTOFF,
    reorder_two_qubit_matrix,
    svd_truncate,
)

logger = logging.getLogger("qudenoise")

_DENSE_WARN_QUBITS = 20
_DENSE_MAX_QUBITS = 24


class MPS:
    """Matrix product state of ``n_qubits`` qubits.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    bond_dim:
        Hard cap ``chi`` on every bond dimension (``None`` = no cap).
    truncation_threshold:
        Adaptive truncation: singular values below this value are discarded
        (singular values are normalized so that their squares sum to one).
        May be combined with ``bond_dim``: the threshold triggers, ``bond_dim``
        caps.
    init_state:
        Computational basis product state, e.g. ``"0101"`` (default: all zeros).
    device:
        ``"auto"`` (CuPy if available, else NumPy), ``"cpu"``/``"numpy"``, or
        ``"cuda"``/``"gpu"``/``"cupy"``. Forwarded to
        :func:`qudenoise.backend.configure`; all tensors live on that device.
    svd_cutoff:
        Singular values below this (normalized) are always dropped as
        numerical zeros.
    """

    def __init__(
        self,
        n_qubits: int,
        bond_dim: Optional[int] = None,
        truncation_threshold: Optional[float] = None,
        init_state: Union[str, Sequence[int], None] = None,
        device: str = "auto",
        svd_cutoff: float = DEFAULT_SVD_CUTOFF,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        configure(device, log=False)
        bits = self._parse_init_state(init_state, n_qubits)
        xp = get_backend()
        tensors = []
        for b in bits:
            t = xp.zeros((1, 2, 1), dtype=DTYPE)
            t[0, b, 0] = 1.0
            tensors.append(t)
        self._setup(tensors, 0, bond_dim, truncation_threshold, svd_cutoff)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_init_state(init_state: Union[str, Sequence[int], None], n: int) -> List[int]:
        if init_state is None:
            return [0] * n
        bits = [int(c) for c in init_state]
        if len(bits) != n or any(b not in (0, 1) for b in bits):
            raise ValueError(f"init_state must be a length-{n} string/sequence of 0/1")
        return bits

    def _setup(
        self,
        tensors: List[Any],
        center: int,
        bond_dim: Optional[int],
        threshold: Optional[float],
        cutoff: float,
    ) -> None:
        if bond_dim is not None and bond_dim < 1:
            raise ValueError("bond_dim must be >= 1")
        if threshold is not None and threshold < 0:
            raise ValueError("truncation_threshold must be >= 0")
        self.tensors: List[Any] = tensors
        self.n_qubits: int = len(tensors)
        self.center: int = center
        self.bond_dim = bond_dim
        self.truncation_threshold = threshold
        self.svd_cutoff = cutoff
        # truncation bookkeeping
        self.truncation_error: float = 0.0
        self.n_truncations: int = 0
        self._log_fidelity: float = 0.0
        self._sum_sqrt_error: float = 0.0
        self.truncation_history: List[Tuple[int, float]] = []  # (bond index, eps)

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[Any],
        bond_dim: Optional[int] = None,
        truncation_threshold: Optional[float] = None,
        svd_cutoff: float = DEFAULT_SVD_CUTOFF,
    ) -> "MPS":
        """Build an MPS from raw site tensors (moved to the active backend)."""
        tens = [to_backend(t, DTYPE) for t in tensors]
        if not tens:
            raise ValueError("Need at least one tensor")
        for i, t in enumerate(tens):
            if t.ndim != 3 or t.shape[1] != 2:
                raise ValueError(f"Tensor {i} must have shape (Dl, 2, Dr), got {t.shape}")
            if i > 0 and tens[i - 1].shape[2] != t.shape[0]:
                raise ValueError(f"Bond mismatch between tensors {i - 1} and {i}")
        if tens[0].shape[0] != 1 or tens[-1].shape[2] != 1:
            raise ValueError("Boundary bond dimensions must be 1")
        obj = cls.__new__(cls)
        obj._setup(tens, 0, bond_dim, truncation_threshold, svd_cutoff)
        obj.center = obj.n_qubits - 1  # will be established by the sweep below
        obj._left_orthogonalize_all()
        return obj

    @classmethod
    def from_dense(
        cls,
        vector: Any,
        bond_dim: Optional[int] = None,
        truncation_threshold: Optional[float] = None,
        svd_cutoff: float = DEFAULT_SVD_CUTOFF,
        normalize: bool = True,
    ) -> "MPS":
        """Decompose a dense statevector (length ``2**n``, big-endian) into an MPS
        by successive SVDs. Exact unless ``bond_dim``/``truncation_threshold``
        force truncation (which is recorded in the truncation bookkeeping).
        """
        vec = to_backend(vector, DTYPE).reshape(-1)
        n = int(round(math.log2(vec.shape[0])))
        if 2**n != vec.shape[0]:
            raise ValueError("Statevector length must be a power of two")
        xp = get_backend()
        if normalize:
            nrm = float(xp.linalg.norm(vec))
            if nrm == 0.0:
                raise ValueError("Cannot build an MPS from the zero vector")
            vec = vec / nrm
        obj = cls.__new__(cls)
        obj._setup([], n - 1, bond_dim, truncation_threshold, svd_cutoff)
        tensors: List[Any] = []
        psi = vec.reshape(1, -1)
        for i in range(n - 1):
            d_left = psi.shape[0]
            u, s, vh, disc = svd_truncate(
                psi.reshape(d_left * 2, -1), bond_dim, truncation_threshold, svd_cutoff
            )
            if disc > 0.0:
                obj._record_truncation(i, disc)
            tensors.append(u.reshape(d_left, 2, -1))
            psi = s[:, None] * vh
        tensors.append(psi.reshape(psi.shape[0], 2, 1))
        obj.tensors = tensors
        obj.n_qubits = n
        obj.center = n - 1
        return obj

    def copy(self) -> "MPS":
        """Deep copy (tensors, canonical center, truncation bookkeeping)."""
        obj = MPS.__new__(MPS)
        obj._setup(
            [t.copy() for t in self.tensors],
            self.center,
            self.bond_dim,
            self.truncation_threshold,
            self.svd_cutoff,
        )
        obj.truncation_error = self.truncation_error
        obj.n_truncations = self.n_truncations
        obj._log_fidelity = self._log_fidelity
        obj._sum_sqrt_error = self._sum_sqrt_error
        obj.truncation_history = list(self.truncation_history)
        return obj

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_qubits

    def __repr__(self) -> str:
        return (
            f"MPS(n_qubits={self.n_qubits}, bond_dims={self.bond_dimensions()}, "
            f"chi_max={self.bond_dim}, threshold={self.truncation_threshold}, "
            f"center={self.center}, truncation_error={self.truncation_error:.3e})"
        )

    def bond_dimensions(self) -> List[int]:
        """Dimensions of the ``n_qubits - 1`` internal bonds."""
        return [int(t.shape[2]) for t in self.tensors[:-1]]

    @property
    def max_bond_dimension(self) -> int:
        dims = self.bond_dimensions()
        return max(dims) if dims else 1

    @property
    def fidelity_estimate(self) -> float:
        """``prod (1 - eps_k)``: estimated fidelity with the untruncated state."""
        return math.exp(self._log_fidelity)

    @property
    def fidelity_lower_bound(self) -> float:
        """Rigorous lower bound ``1 - (sum sqrt(eps_k))**2`` on that fidelity."""
        return max(0.0, 1.0 - self._sum_sqrt_error**2)

    def _record_truncation(self, bond: int, discarded: float) -> None:
        self.truncation_error += discarded
        self.n_truncations += 1
        self._log_fidelity += math.log1p(-min(discarded, 1.0 - 1e-300))
        self._sum_sqrt_error += math.sqrt(discarded)
        self.truncation_history.append((bond, discarded))

    def reset_truncation_error(self) -> None:
        self.truncation_error = 0.0
        self.n_truncations = 0
        self._log_fidelity = 0.0
        self._sum_sqrt_error = 0.0
        self.truncation_history = []

    # ------------------------------------------------------------------
    # canonical form
    # ------------------------------------------------------------------
    def _shift_right(self, i: int) -> None:
        """Move the center from site ``i`` to ``i + 1`` (QR)."""
        xp = get_backend()
        t = self.tensors[i]
        dl, _, dr = t.shape
        q, r = xp.linalg.qr(t.reshape(dl * 2, dr))
        self.tensors[i] = q.reshape(dl, 2, q.shape[1])
        self.tensors[i + 1] = xp.einsum("ab,bsc->asc", r, self.tensors[i + 1])

    def _shift_left(self, i: int) -> None:
        """Move the center from site ``i`` to ``i - 1`` (LQ via QR of the adjoint)."""
        xp = get_backend()
        t = self.tensors[i]
        dl, _, dr = t.shape
        q, r = xp.linalg.qr(t.reshape(dl, 2 * dr).conj().T)  # t = r^dagger q^dagger
        self.tensors[i] = q.conj().T.reshape(q.shape[1], 2, dr)
        self.tensors[i - 1] = xp.einsum("lsm,mk->lsk", self.tensors[i - 1], r.conj().T)

    def _left_orthogonalize_all(self) -> None:
        for i in range(self.n_qubits - 1):
            self._shift_right(i)
        self.center = self.n_qubits - 1

    def canonicalize(self, center: int) -> "MPS":
        """Move the orthogonality center to ``center`` via QR/LQ sweeps."""
        if not 0 <= center < self.n_qubits:
            raise ValueError(f"center {center} out of range for {self.n_qubits} qubits")
        while self.center < center:
            self._shift_right(self.center)
            self.center += 1
        while self.center > center:
            self._shift_left(self.center)
            self.center -= 1
        return self

    def norm(self) -> float:
        """2-norm of the state (O(chi^2) thanks to the canonical center)."""
        xp = get_backend()
        return float(xp.linalg.norm(self.tensors[self.center]))

    def normalize(self) -> "MPS":
        nrm = self.norm()
        if not nrm > 0.0:
            raise ValueError("Cannot normalize a zero-norm state")
        self.tensors[self.center] = self.tensors[self.center] / nrm
        return self

    # ------------------------------------------------------------------
    # gate application (unitary; truncating for 2-qubit gates)
    # ------------------------------------------------------------------
    def apply_1q(self, matrix: Any, qubit: int) -> None:
        """Apply a single-qubit operator: one ``einsum`` on the physical index.

        Assumes a unitary. (Non-unitary operators must first move the center to
        ``qubit`` -- see :mod:`qudenoise.noise` -- to keep the canonical form
        consistent.)
        """
        xp = get_backend()
        self.tensors[qubit] = xp.einsum("ts,lsr->ltr", matrix, self.tensors[qubit])

    def prepare_two_site(self, i: int) -> None:
        """Ensure the orthogonality center sits on site ``i`` or ``i + 1``."""
        if self.center in (i, i + 1):
            return
        self.canonicalize(i if self.center < i else i + 1)

    def two_site_tensor(self, i: int) -> Any:
        """Contract sites ``i, i+1`` into ``theta[l, s_i, s_{i+1}, r]``."""
        xp = get_backend()
        return xp.einsum("lsm,mtr->lstr", self.tensors[i], self.tensors[i + 1])

    def set_two_site_tensor(self, i: int, theta: Any) -> float:
        """SVD-split ``theta`` back onto sites ``i, i+1`` with truncation.

        The singular values are absorbed into site ``i + 1`` (the new center).
        Returns the discarded probability weight of this truncation.
        """
        dl, _, _, dr = theta.shape
        u, s, vh, disc = svd_truncate(
            theta.reshape(dl * 2, 2 * dr),
            self.bond_dim,
            self.truncation_threshold,
            self.svd_cutoff,
        )
        k = s.shape[0]
        self.tensors[i] = u.reshape(dl, 2, k)
        self.tensors[i + 1] = (s[:, None] * vh).reshape(k, 2, dr)
        self.center = i + 1
        if disc > 0.0:
            self._record_truncation(i, disc)
        return disc

    def apply_2q(self, matrix: Any, qubit0: int, qubit1: int) -> float:
        """Apply a two-qubit unitary on *adjacent* qubits.

        ``matrix`` is ``4x4`` in the basis ``|q0 q1>`` for the qubit order given
        (either ``(i, i+1)`` or ``(i+1, i)`` is accepted). Merge -> contract ->
        reshape -> SVD -> truncate -> split. Returns the discarded weight.
        """
        if abs(qubit0 - qubit1) != 1:
            raise ValueError(
                f"apply_2q needs adjacent qubits, got ({qubit0}, {qubit1}); "
                "use Circuit/Simulator for automatic SWAP routing"
            )
        xp = get_backend()
        if qubit0 > qubit1:
            matrix = reorder_two_qubit_matrix(matrix)
        i = min(qubit0, qubit1)
        self.prepare_two_site(i)
        theta = xp.einsum("abst,lstr->labr", matrix.reshape(2, 2, 2, 2), self.two_site_tensor(i))
        return self.set_two_site_tensor(i, theta)

    def apply_gate(self, matrix: Any, qubits: Sequence[int]) -> float:
        """Dispatch on the number of qubits (1 or 2). Returns discarded weight."""
        if len(qubits) == 1:
            self.apply_1q(matrix, qubits[0])
            return 0.0
        if len(qubits) == 2:
            return self.apply_2q(matrix, qubits[0], qubits[1])
        raise ValueError("Only 1- and 2-qubit gates are supported")

    # ------------------------------------------------------------------
    # conversions & structural utilities
    # ------------------------------------------------------------------
    def to_dense(self, force: bool = False) -> Any:
        """Contract to a dense statevector of length ``2**n`` (big-endian).

        For validation only: guarded because memory grows as ``2**n``. Warns
        above 20 qubits and refuses above 24 unless ``force=True``.
        """
        n = self.n_qubits
        if n > _DENSE_MAX_QUBITS and not force:
            raise ValueError(
                f"to_dense() on {n} qubits needs 2**{n} amplitudes; pass force=True to override"
            )
        if n > _DENSE_WARN_QUBITS:
            warnings.warn(f"to_dense() on {n} qubits allocates 2**{n} amplitudes", stacklevel=2)
        xp = get_backend()
        psi = self.tensors[0].reshape(2, -1)
        for t in self.tensors[1:]:
            psi = xp.einsum("pd,dsr->psr", psi, t)
            psi = psi.reshape(psi.shape[0] * 2, psi.shape[2])
        return psi.reshape(-1)

    def project_and_drop_trailing(self, n_keep: int) -> Tuple["MPS", float]:
        """Project qubits ``n_keep .. n-1`` onto ``|0>`` and remove them.

        Returns ``(new_mps, probability)`` where ``probability`` is the
        probability of finding all dropped qubits in ``|0>`` (relative to the
        current norm) and ``new_mps`` is the normalized post-measurement state
        of the first ``n_keep`` qubits.
        """
        n = self.n_qubits
        if not 1 <= n_keep < n:
            raise ValueError(f"n_keep must be in [1, {n - 1}]")
        xp = get_backend()
        old_norm_sq = self.norm() ** 2
        chain = self.tensors[n - 1][:, 0, :]
        for k in range(n - 2, n_keep - 1, -1):
            chain = self.tensors[k][:, 0, :] @ chain
        tensors = [t.copy() for t in self.tensors[: n_keep - 1]]
        tensors.append(xp.einsum("lsm,mr->lsr", self.tensors[n_keep - 1], chain))
        new = MPS.from_tensors(
            tensors, self.bond_dim, self.truncation_threshold, self.svd_cutoff
        )
        weight = new.norm() ** 2
        if not weight > 1e-300:
            raise ValueError("Projection onto |0...0> has zero probability")
        new.normalize()
        return new, weight / old_norm_sq

    def extend_with_zeros(self, n_extra: int) -> "MPS":
        """Return a copy with ``n_extra`` extra qubits in ``|0>`` appended on the right."""
        xp = get_backend()
        new = self.copy()
        for _ in range(n_extra):
            t = xp.zeros((1, 2, 1), dtype=DTYPE)
            t[0, 0, 0] = 1.0
            new.tensors.append(t)
        new.n_qubits += n_extra
        return new

    def to_host_tensors(self) -> List[Any]:
        """Site tensors as host NumPy arrays (for plotting / serialization)."""
        from .backend import to_numpy

        return [to_numpy(t) for t in self.tensors]
