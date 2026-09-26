"""Gate library.

Every gate is a plain *function* returning a backend array (``complex128``),
built on whichever backend is active at call time. Nothing is cached across
backends, and there are no gate classes, so the functions stay trivially
portable between devices.

Conventions
-----------
* Qubit ordering is big-endian: for a two-qubit gate on ``(q0, q1)`` the
  matrix basis order is ``|q0 q1>`` = ``|00>, |01>, |10>, |11>``, so
  ``cnot`` has ``q0`` as control and ``q1`` as target.
* Rotations are ``exp(-i * theta * P / 2)`` (``P`` a Pauli string), which
  makes them compatible with the parameter-shift rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict

from .backend import DTYPE, get_backend

_SQRT1_2 = 1.0 / math.sqrt(2.0)


def _arr(nested: Any) -> Any:
    return get_backend().array(nested, dtype=DTYPE)


# --------------------------------------------------------------------------
# Fixed single-qubit gates
# --------------------------------------------------------------------------
def I() -> Any:  # noqa: E743 - conventional name
    return _arr([[1, 0], [0, 1]])


def X() -> Any:
    return _arr([[0, 1], [1, 0]])


def Y() -> Any:
    return _arr([[0, -1j], [1j, 0]])


def Z() -> Any:
    return _arr([[1, 0], [0, -1]])


def H() -> Any:
    return _arr([[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]])


def S() -> Any:
    return _arr([[1, 0], [0, 1j]])


def Sdg() -> Any:
    return _arr([[1, 0], [0, -1j]])


def T() -> Any:
    return _arr([[1, 0], [0, complex(_SQRT1_2, _SQRT1_2)]])


def Tdg() -> Any:
    return _arr([[1, 0], [0, complex(_SQRT1_2, -_SQRT1_2)]])


# --------------------------------------------------------------------------
# Fixed two-qubit gates
# --------------------------------------------------------------------------
def CNOT() -> Any:
    return _arr([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def CZ() -> Any:
    return _arr([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


def SWAP() -> Any:
    return _arr([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


# --------------------------------------------------------------------------
# Parameterized gates
# --------------------------------------------------------------------------
def RX(theta: float) -> Any:
    c, s = math.cos(float(theta) / 2), math.sin(float(theta) / 2)
    return _arr([[c, -1j * s], [-1j * s, c]])


def RY(theta: float) -> Any:
    c, s = math.cos(float(theta) / 2), math.sin(float(theta) / 2)
    return _arr([[c, -s], [s, c]])


def RZ(theta: float) -> Any:
    h = float(theta) / 2
    return _arr([[complex(math.cos(h), -math.sin(h)), 0], [0, complex(math.cos(h), math.sin(h))]])


def P(phi: float) -> Any:
    """Phase gate ``diag(1, e^{i phi})``."""
    return _arr([[1, 0], [0, complex(math.cos(float(phi)), math.sin(float(phi)))]])


def _pauli_pair_rotation(pauli: Any, theta: float) -> Any:
    """``exp(-i theta/2 * P (x) P) = cos(theta/2) I - i sin(theta/2) P (x) P``."""
    xp = get_backend()
    c, s = math.cos(float(theta) / 2), math.sin(float(theta) / 2)
    return c * xp.eye(4, dtype=DTYPE) - 1j * s * xp.kron(pauli, pauli)


def RXX(theta: float) -> Any:
    return _pauli_pair_rotation(X(), theta)


def RYY(theta: float) -> Any:
    return _pauli_pair_rotation(Y(), theta)


def RZZ(theta: float) -> Any:
    return _pauli_pair_rotation(Z(), theta)


def entangler(a: float, b: float, c: float) -> Any:
    """Generic parameterized two-qubit entangler ``exp(-i/2 (a XX + b YY + c ZZ))``.

    The three terms commute, so this factorizes exactly into ``RXX(a) RYY(b)
    RZZ(c)``. Together with single-qubit rotations it spans all two-qubit
    unitaries (the Cartan/KAK decomposition), and each angle obeys the
    parameter-shift rule independently.
    """
    return RXX(a) @ RYY(b) @ RZZ(c)


# --------------------------------------------------------------------------
# Registry (used by Circuit and the CLI)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GateSpec:
    name: str
    n_qubits: int
    n_params: int
    factory: Callable[..., Any]


GATES: Dict[str, GateSpec] = {
    spec.name: spec
    for spec in [
        GateSpec("i", 1, 0, I),
        GateSpec("x", 1, 0, X),
        GateSpec("y", 1, 0, Y),
        GateSpec("z", 1, 0, Z),
        GateSpec("h", 1, 0, H),
        GateSpec("s", 1, 0, S),
        GateSpec("sdg", 1, 0, Sdg),
        GateSpec("t", 1, 0, T),
        GateSpec("tdg", 1, 0, Tdg),
        GateSpec("rx", 1, 1, RX),
        GateSpec("ry", 1, 1, RY),
        GateSpec("rz", 1, 1, RZ),
        GateSpec("p", 1, 1, P),
        GateSpec("cnot", 2, 0, CNOT),
        GateSpec("cz", 2, 0, CZ),
        GateSpec("swap", 2, 0, SWAP),
        GateSpec("rxx", 2, 1, RXX),
        GateSpec("ryy", 2, 1, RYY),
        GateSpec("rzz", 2, 1, RZZ),
        GateSpec("entangler", 2, 3, entangler),
    ]
}
GATE_ALIASES = {"cx": "cnot", "id": "i", "phase": "p"}


def canonical_name(name: str) -> str:
    key = name.lower()
    return GATE_ALIASES.get(key, key)


def get_gate(name: str, *params: float) -> Any:
    """Build gate ``name`` with the given parameters on the active backend."""
    key = canonical_name(name)
    try:
        spec = GATES[key]
    except KeyError:
        raise ValueError(f"Unknown gate {name!r}. Known gates: {sorted(GATES)}") from None
    if len(params) != spec.n_params:
        raise ValueError(f"Gate {key!r} takes {spec.n_params} parameter(s), got {len(params)}")
    return spec.factory(*params)
