"""Dense reference simulators, for validation in tests (not a product feature).

* :class:`DenseSimulator`         -- exact statevector, pure circuits (any qubit pairs).
* :class:`DensityMatrixSimulator` -- exact density matrix, including noise channels.

Both use the same conventions as the MPS code (big-endian qubit order, same
gate matrices) and go through :mod:`qudenoise.backend`, so they run on
whichever backend is active. Practical limit: ~12 qubits (statevector) and
~6 qubits (density matrix).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .backend import DTYPE, get_backend
from .circuit import Circuit

_MAX_QUBITS = 14


def _apply_matrix(tensor: Any, matrix: Any, axes: Sequence[int]) -> Any:
    """Apply ``matrix`` (2^k x 2^k) to the given tensor axes."""
    xp = get_backend()
    k = len(axes)
    op = matrix.reshape((2,) * (2 * k))
    res = xp.tensordot(op, tensor, axes=(list(range(k, 2 * k)), list(axes)))
    return xp.moveaxis(res, list(range(k)), list(axes))


class DenseSimulator:
    """Exact ``2**n`` statevector simulator."""

    def __init__(self, n_qubits: int, init_state: Optional[str] = None) -> None:
        if n_qubits > _MAX_QUBITS:
            raise ValueError(f"Dense reference limited to {_MAX_QUBITS} qubits")
        xp = get_backend()
        self.n_qubits = n_qubits
        bits = [0] * n_qubits if init_state is None else [int(c) for c in init_state]
        psi = xp.zeros((2,) * n_qubits, dtype=DTYPE)
        psi[tuple(bits)] = 1.0
        self.psi = psi

    def apply(self, matrix: Any, qubits: Sequence[int]) -> None:
        self.psi = _apply_matrix(self.psi, matrix, qubits)

    def run(self, circuit: Circuit) -> Any:
        """Execute ``circuit`` (no routing needed) and return the flat statevector."""
        if circuit.n_qubits != self.n_qubits:
            raise ValueError("Qubit count mismatch")
        for op in circuit.compile(route=False):
            if op.op_type != "gate":
                raise ValueError("DenseSimulator is noiseless; use DensityMatrixSimulator")
            self.apply(op.data, op.qubits)
        return self.statevector()

    def statevector(self) -> Any:
        return self.psi.reshape(-1)

    def expectation(self, operators: Mapping[int, Any]) -> complex:
        """``<psi| prod_q O_q |psi>`` for single-qubit operators."""
        xp = get_backend()
        phi = self.psi
        for q, op in operators.items():
            phi = _apply_matrix(phi, op, [q])
        return complex(xp.vdot(self.psi, phi))


class DensityMatrixSimulator:
    """Exact density-matrix simulator with Kraus channels."""

    def __init__(self, n_qubits: int, init_state: Optional[str] = None) -> None:
        if n_qubits > 8:
            raise ValueError("Density-matrix reference limited to 8 qubits")
        xp = get_backend()
        self.n_qubits = n_qubits
        bits = [0] * n_qubits if init_state is None else [int(c) for c in init_state]
        rho = xp.zeros((2,) * (2 * n_qubits), dtype=DTYPE)
        rho[tuple(bits) + tuple(bits)] = 1.0
        self.rho = rho

    def apply_unitary(self, matrix: Any, qubits: Sequence[int]) -> None:
        n = self.n_qubits
        self.rho = _apply_matrix(self.rho, matrix, list(qubits))
        self.rho = _apply_matrix(self.rho, matrix.conj(), [n + q for q in qubits])

    def run(self, circuit: Circuit) -> Any:
        if circuit.n_qubits != self.n_qubits:
            raise ValueError("Qubit count mismatch")
        for op in circuit.compile(route=False):
            if op.op_type == "gate":
                self.apply_unitary(op.data, op.qubits)
            else:
                self.rho = op.data.apply_to_density_matrix(self.rho, op.qubits, self.n_qubits)
        return self.matrix()

    def matrix(self) -> Any:
        d = 2**self.n_qubits
        return self.rho.reshape(d, d)

    def expectation(self, operators: Mapping[int, Any]) -> complex:
        """``tr(rho prod_q O_q)``."""
        xp = get_backend()
        rho = self.rho
        for q, op in operators.items():
            rho = _apply_matrix(rho, op, [q])
        d = 2**self.n_qubits
        return complex(xp.trace(rho.reshape(d, d)))
