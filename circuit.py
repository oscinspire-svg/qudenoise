"""Circuit builder and compiler.

A :class:`Circuit` is a lightweight, backend-independent list of instructions.
:meth:`Circuit.compile` turns it into a flat, inspectable list of
``Op(op_type, qubits, data, name)`` records that the simulator consumes:

* ``op_type == "gate"``  -> ``data`` is a matrix (``2x2`` or ``4x4``) on the
  active backend, ``qubits`` has length 1 or 2.
* ``op_type == "noise"`` -> ``data`` is a :class:`~qudenoise.noise.KrausChannel`.

Non-adjacent two-qubit operations are made adjacent by wrapping them in SWAP
chains. Every insertion is logged (``logging`` INFO on the ``qudenoise``
logger) and recorded in :attr:`Circuit.routing_log`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from . import gates as G
from . import noise as N
from .backend import DTYPE, to_backend, to_numpy

logger = logging.getLogger("qudenoise")


class Op(NamedTuple):
    """One compiled operation."""

    op_type: str  # "gate" | "noise"
    qubits: Tuple[int, ...]
    data: Any  # matrix (gate) or KrausChannel (noise)
    name: str = ""


@dataclass(frozen=True)
class Instruction:
    kind: str  # "gate" | "noise"
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    payload: Any = None  # host matrix for custom unitaries / KrausChannel for noise


# inverse rules for named gates
_SELF_INVERSE = {"i", "x", "y", "z", "h", "cnot", "cz", "swap"}
_DAGGER_PAIRS = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
_NEGATE_PARAMS = {"rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "entangler"}


class Circuit:
    """Quantum circuit on ``n_qubits`` qubits (qubit 0 = leftmost MPS site)."""

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = int(n_qubits)
        self._instructions: List[Instruction] = []
        self.routing_log: List[str] = []
        self.n_swaps_inserted: int = 0

    # ------------------------------------------------------------------
    # container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._instructions)

    def __iter__(self):
        return iter(self._instructions)

    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self.n_qubits}, n_instructions={len(self)})"

    @property
    def instructions(self) -> List[Instruction]:
        return list(self._instructions)

    @property
    def has_noise(self) -> bool:
        return any(ins.kind == "noise" for ins in self._instructions)

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------
    def _check(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        qs = tuple(int(q) for q in qubits)
        if len(set(qs)) != len(qs):
            raise ValueError(f"Repeated qubit in {qs}")
        for q in qs:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"Qubit {q} out of range for a {self.n_qubits}-qubit circuit")
        return qs

    def add_gate(self, name: str, qubits: Sequence[int], *params: float) -> "Circuit":
        """Append a named gate from :data:`qudenoise.gates.GATES`."""
        key = G.canonical_name(name)
        if key not in G.GATES:
            raise ValueError(f"Unknown gate {name!r}. Known gates: {sorted(G.GATES)}")
        spec = G.GATES[key]
        qs = self._check(qubits)
        if len(qs) != spec.n_qubits:
            raise ValueError(f"Gate {key!r} acts on {spec.n_qubits} qubit(s), got {len(qs)}")
        if len(params) != spec.n_params:
            raise ValueError(f"Gate {key!r} takes {spec.n_params} parameter(s), got {len(params)}")
        self._instructions.append(Instruction("gate", key, qs, tuple(float(p) for p in params)))
        return self

    def unitary(self, matrix: Any, qubits: Sequence[int], name: str = "unitary") -> "Circuit":
        """Append an arbitrary 1- or 2-qubit unitary (columns = input basis, big-endian)."""
        qs = self._check(qubits)
        host = to_numpy(matrix).astype("complex128")
        if len(qs) not in (1, 2) or host.shape != (2 ** len(qs),) * 2:
            raise ValueError("unitary needs a 2x2 (1 qubit) or 4x4 (2 qubit) matrix")
        self._instructions.append(Instruction("gate", name, qs, (), host))
        return self

    def add_noise_channel(
        self,
        channel: Union[str, N.KrausChannel],
        qubits: Sequence[int],
        params: Optional[float] = None,
    ) -> "Circuit":
        """Append a noise channel.

        ``channel`` is either a :class:`~qudenoise.noise.KrausChannel` or a
        name (``"depolarizing"``, ``"amplitude_damping"``, ``"phase_damping"``,
        ``"bit_flip"``, ``"phase_flip"``) together with its error
        probability/rate in ``params``.
        """
        qs = self._check(qubits)
        if isinstance(channel, str):
            if params is None:
                raise ValueError(f"Channel {channel!r} needs its error rate in `params`")
            ch = N.make_channel(channel, float(params), n_qubits=len(qs))
        else:
            ch = channel
        if ch.n_qubits != len(qs):
            raise ValueError(f"{ch.name} acts on {ch.n_qubits} qubit(s), got {len(qs)}")
        self._instructions.append(Instruction("noise", ch.name, qs, (), ch))
        return self

    # single-qubit gates
    def i(self, q: int) -> "Circuit":
        return self.add_gate("i", [q])

    def x(self, q: int) -> "Circuit":
        return self.add_gate("x", [q])

    def y(self, q: int) -> "Circuit":
        return self.add_gate("y", [q])

    def z(self, q: int) -> "Circuit":
        return self.add_gate("z", [q])

    def h(self, q: int) -> "Circuit":
        return self.add_gate("h", [q])

    def s(self, q: int) -> "Circuit":
        return self.add_gate("s", [q])

    def sdg(self, q: int) -> "Circuit":
        return self.add_gate("sdg", [q])

    def t(self, q: int) -> "Circuit":
        return self.add_gate("t", [q])

    def tdg(self, q: int) -> "Circuit":
        return self.add_gate("tdg", [q])

    def rx(self, q: int, theta: float) -> "Circuit":
        return self.add_gate("rx", [q], theta)

    def ry(self, q: int, theta: float) -> "Circuit":
        return self.add_gate("ry", [q], theta)

    def rz(self, q: int, theta: float) -> "Circuit":
        return self.add_gate("rz", [q], theta)

    def p(self, q: int, phi: float) -> "Circuit":
        return self.add_gate("p", [q], phi)

    # two-qubit gates
    def cnot(self, control: int, target: int) -> "Circuit":
        return self.add_gate("cnot", [control, target])

    cx = cnot

    def cz(self, a: int, b: int) -> "Circuit":
        return self.add_gate("cz", [a, b])

    def swap(self, a: int, b: int) -> "Circuit":
        return self.add_gate("swap", [a, b])

    def rxx(self, a: int, b: int, theta: float) -> "Circuit":
        return self.add_gate("rxx", [a, b], theta)

    def ryy(self, a: int, b: int, theta: float) -> "Circuit":
        return self.add_gate("ryy", [a, b], theta)

    def rzz(self, a: int, b: int, theta: float) -> "Circuit":
        return self.add_gate("rzz", [a, b], theta)

    def entangler(self, a: int, b: int, ax: float, ay: float, az: float) -> "Circuit":
        return self.add_gate("entangler", [a, b], ax, ay, az)

    # noise shortcuts
    def depolarizing(self, qubits: Union[int, Sequence[int]], p: float) -> "Circuit":
        return self.add_noise_channel("depolarizing", _as_tuple(qubits), p)

    def amplitude_damping(self, q: int, gamma: float) -> "Circuit":
        return self.add_noise_channel("amplitude_damping", [q], gamma)

    def phase_damping(self, q: int, lam: float) -> "Circuit":
        return self.add_noise_channel("phase_damping", [q], lam)

    def bit_flip(self, q: int, p: float) -> "Circuit":
        return self.add_noise_channel("bit_flip", [q], p)

    def phase_flip(self, q: int, p: float) -> "Circuit":
        return self.add_noise_channel("phase_flip", [q], p)

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------
    def extend(self, other: "Circuit") -> "Circuit":
        """Append all instructions of ``other`` (same qubit count required)."""
        if other.n_qubits != self.n_qubits:
            raise ValueError("Circuits act on different numbers of qubits")
        self._instructions.extend(other._instructions)
        return self

    def copy(self) -> "Circuit":
        new = Circuit(self.n_qubits)
        new._instructions = list(self._instructions)
        return new

    def inverse(self) -> "Circuit":
        """The adjoint circuit (reversed order, inverted gates). Noise is not invertible."""
        inv = Circuit(self.n_qubits)
        for ins in reversed(self._instructions):
            if ins.kind == "noise":
                raise ValueError("Cannot invert a circuit containing noise channels")
            if ins.payload is not None:
                inv._instructions.append(
                    Instruction("gate", ins.name + "^dag", ins.qubits, (), ins.payload.conj().T)
                )
            elif ins.name in _SELF_INVERSE:
                inv._instructions.append(ins)
            elif ins.name in _DAGGER_PAIRS:
                inv._instructions.append(Instruction("gate", _DAGGER_PAIRS[ins.name], ins.qubits))
            elif ins.name in _NEGATE_PARAMS:
                inv._instructions.append(
                    Instruction("gate", ins.name, ins.qubits, tuple(-p for p in ins.params))
                )
            else:  # pragma: no cover - registry/inverse tables out of sync
                raise ValueError(f"No inverse rule for gate {ins.name!r}")
        return inv

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------
    def compile(self, route: bool = True) -> List[Op]:
        """Compile to an ordered list of :class:`Op` on the active backend.

        With ``route=True`` non-adjacent two-qubit operations are wrapped in SWAP
        chains (logged, and recorded in ``self.routing_log``). With
        ``route=False`` they are passed through unchanged (used by the dense
        reference simulator, which needs no routing).
        """
        self.routing_log = []
        self.n_swaps_inserted = 0
        ops: List[Op] = []
        swap = None
        for ins in self._instructions:
            if ins.kind == "gate":
                if ins.payload is not None:
                    data = to_backend(ins.payload, DTYPE)
                else:
                    data = G.get_gate(ins.name, *ins.params)
            else:
                data = ins.payload
            op = Op(ins.kind, ins.qubits, data, ins.name)
            if len(ins.qubits) == 2 and abs(ins.qubits[0] - ins.qubits[1]) > 1 and route:
                if swap is None:
                    swap = G.SWAP()
                ops.extend(self._route(op, swap))
            else:
                ops.append(op)
        return ops

    def _route(self, op: Op, swap: Any) -> List[Op]:
        a, b = op.qubits
        d = abs(a - b)
        step = 1 if a < b else -1
        # move qubit `a` next to `b`
        path = [(a + step * k, a + step * (k + 1)) for k in range(d - 1)]
        target = (b - step, b)
        n_new = 2 * len(path)
        msg = (
            f"Routing non-adjacent {op.name}{op.qubits}: inserting {n_new} SWAPs "
            f"(qubit {a} -> position {b - step}, then back)"
        )
        logger.info(msg)
        self.routing_log.append(msg)
        self.n_swaps_inserted += n_new
        out = [Op("gate", pair, swap, "swap") for pair in path]
        out.append(Op(op.op_type, target, op.data, op.name))
        out.extend(Op("gate", pair, swap, "swap") for pair in reversed(path))
        return out

    # ------------------------------------------------------------------
    # (de)serialization for the CLI
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        ops: List[Dict[str, Any]] = []
        for ins in self._instructions:
            if ins.kind == "gate":
                if ins.payload is not None:
                    raise ValueError("Custom unitaries are not JSON-serializable")
                ops.append({"gate": ins.name, "qubits": list(ins.qubits), "params": list(ins.params)})
            else:
                ch = ins.payload
                if len(ch.params) != 1:
                    raise ValueError("Only named single-parameter channels are serializable")
                ops.append(
                    {
                        "noise": ins.name,
                        "qubits": list(ins.qubits),
                        "param": next(iter(ch.params.values())),
                    }
                )
        return {"n_qubits": self.n_qubits, "ops": ops}

    @classmethod
    def from_dict(cls, spec: Dict[str, Any], n_qubits: Optional[int] = None) -> "Circuit":
        """Build a circuit from ``{"n_qubits": N, "ops": [...]}``.

        Each op is ``{"gate": name, "qubits": [...], "params": [...]}`` or
        ``{"noise": name, "qubits": [...], "param": rate}`` (``p`` / ``gamma`` /
        ``lam`` are accepted as synonyms of ``param``).
        """
        n = n_qubits if n_qubits is not None else spec.get("n_qubits")
        if n is None:
            raise ValueError("Circuit spec has no 'n_qubits' and none was supplied")
        circ = cls(int(n))
        for i, op in enumerate(spec.get("ops", [])):
            try:
                if "gate" in op:
                    circ.add_gate(op["gate"], op["qubits"], *op.get("params", []))
                elif "noise" in op:
                    rate = next(op[k] for k in ("param", "p", "gamma", "lam") if k in op)
                    circ.add_noise_channel(op["noise"], op["qubits"], rate)
                else:
                    raise ValueError("op needs a 'gate' or 'noise' key")
            except (KeyError, StopIteration, ValueError) as exc:
                raise ValueError(f"Bad circuit op #{i} {op!r}: {exc}") from exc
        return circ


def _as_tuple(qubits: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    return (int(qubits),) if isinstance(qubits, int) else tuple(qubits)
