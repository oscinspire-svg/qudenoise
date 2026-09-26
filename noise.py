"""Kraus noise channels and quantum-trajectory (Monte Carlo wavefunction) sampling.

A channel ``rho -> sum_k K_k rho K_k^dagger`` is unravelled into a stochastic
pure-state evolution: on every noisy location one Kraus operator ``K_k`` is
drawn with probability ``p_k = ||K_k |psi>||^2`` and the state is replaced by
``K_k |psi> / sqrt(p_k)``. Averaging any observable over many trajectories
reproduces the density-matrix result.

For MPS the branch probabilities are *local*: after moving the orthogonality
center onto the affected sites, ``p_k`` is just the squared Frobenius norm of
the locally updated center tensor, so no global contraction is needed.

Fast path: if every ``K_k^dagger K_k`` is proportional to the identity (Pauli
channels: depolarizing, bit-flip, phase-flip) the probabilities are
state-independent, so a Kraus operator is drawn up front and applied as a
plain (rescaled) unitary gate, with no canonicalization at all for 1 qubit.

Randomness comes exclusively from an explicitly passed generator
(``backend.get_rng(seed)``); there is no global RNG state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import gates as G
from .backend import DTYPE, backend_name, draw_uniform, get_backend, to_backend, to_numpy
from .mps import MPS
from .utils import reorder_two_qubit_matrix

_COMPLETENESS_TOL = 1e-10


@dataclass
class _SamplingData:
    kraus: Tuple[Any, ...]
    probs: Optional[List[float]]  # state-independent probs if scaled-unitary
    unitaries: Optional[Tuple[Any, ...]]  # K_k / sqrt(p_k) if scaled-unitary


class KrausChannel:
    """A completely-positive trace-preserving map given by Kraus operators.

    The operators are stored on the host and moved to the active backend on
    demand (cached per backend), so a channel object can be built once and
    used with any device.
    """

    def __init__(
        self,
        name: str,
        kraus: Sequence[Any],
        params: Optional[Dict[str, float]] = None,
        validate: bool = True,
    ) -> None:
        host = tuple(to_numpy(k).astype("complex128") for k in kraus)
        if not host:
            raise ValueError("A channel needs at least one Kraus operator")
        dim = host[0].shape[0]
        n_qubits = int(round(math.log2(dim)))
        if 2**n_qubits != dim or any(k.shape != (dim, dim) for k in host):
            raise ValueError("Kraus operators must be square with dimension 2**k")
        self.name = name
        self.params = dict(params or {})
        self.n_qubits = n_qubits
        self._host = host
        self._cache: Dict[str, _SamplingData] = {}
        if validate:
            err = self.completeness_error()
            if err > _COMPLETENESS_TOL:
                raise ValueError(
                    f"Kraus operators of {name!r} are not trace preserving "
                    f"(||sum K^dag K - I|| = {err:.2e})"
                )

    # pickling: drop device-specific cache
    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"KrausChannel({self.name}, {args}; {len(self._host)} operators on {self.n_qubits}q)"

    def __len__(self) -> int:
        return len(self._host)

    def kraus_ops(self) -> Tuple[Any, ...]:
        """Kraus operators as arrays on the active backend."""
        return self._sampling_data().kraus

    def completeness_error(self) -> float:
        """``max |sum_k K_k^dagger K_k - I|``."""
        xp = get_backend()
        ops = [to_backend(k, DTYPE) for k in self._host]
        total = sum(k.conj().T @ k for k in ops)
        return float(abs(total - xp.eye(ops[0].shape[0], dtype=DTYPE)).max())

    @property
    def is_scaled_unitary(self) -> bool:
        """True iff each ``K_k`` is proportional to a unitary (Pauli-type channels)."""
        return self._sampling_data().probs is not None

    def _sampling_data(self) -> _SamplingData:
        key = backend_name()
        data = self._cache.get(key)
        if data is None:
            xp = get_backend()
            kraus = tuple(to_backend(k, DTYPE) for k in self._host)
            dim = kraus[0].shape[0]
            eye = xp.eye(dim, dtype=DTYPE)
            probs: List[float] = []
            units: List[Any] = []
            scaled = True
            for k in kraus:
                kk = k.conj().T @ k
                c = float(xp.trace(kk).real) / dim
                if float(abs(kk - c * eye).max()) > 1e-12:
                    scaled = False
                    break
                probs.append(c)
                units.append(k / math.sqrt(c) if c > 1e-300 else k)
            data = _SamplingData(kraus, probs if scaled else None, tuple(units) if scaled else None)
            self._cache[key] = data
        return data

    def apply_to_density_matrix(self, rho: Any, qubits: Sequence[int], n_qubits: int) -> Any:
        """Exact action ``sum_k K rho K^dagger`` on a dense density matrix
        (``rho`` has shape ``(2,)*2n``). Used by the dense reference only."""
        xp = get_backend()
        k = len(qubits)
        out = xp.zeros_like(rho)
        for op in self.kraus_ops():
            op_t = op.reshape((2,) * (2 * k))
            tmp = _apply_to_axes(rho, op_t, list(qubits))
            # right multiplication by K^dagger: act with conj(K) on the bra axes
            tmp = _apply_to_axes(tmp, op_t.conj(), [n_qubits + q for q in qubits])
            out = out + tmp
        return out


def _apply_to_axes(t: Any, op: Any, axes: List[int]) -> Any:
    """Contract ``op`` (rank 2k, out-axes first) onto the given axes of ``t``."""
    xp = get_backend()
    k = len(axes)
    res = xp.tensordot(op, t, axes=(list(range(k, 2 * k)), axes))
    # tensordot puts op's output axes first; move them back to ``axes``
    return xp.moveaxis(res, list(range(k)), axes)


# --------------------------------------------------------------------------
# Channel factories
# --------------------------------------------------------------------------
def _check_prob(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def depolarizing(p: float, n_qubits: int = 1) -> KrausChannel:
    """Depolarizing channel.

    1 qubit:  ``rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)``; the Bloch
    vector shrinks by ``1 - 4p/3``.
    2 qubits: ``rho -> (1-p) rho + p/15 sum_{PQ != II} (P(x)Q) rho (P(x)Q)``; the
    Bloch-vector correlators shrink by ``1 - 16p/15``.
    """
    p = _check_prob(p, "p")
    xp = get_backend()
    paulis = [G.I(), G.X(), G.Y(), G.Z()]
    if n_qubits == 1:
        ops = [math.sqrt(1 - p) * paulis[0]] + [math.sqrt(p / 3) * m for m in paulis[1:]]
    elif n_qubits == 2:
        ops = [math.sqrt(1 - p) * xp.kron(paulis[0], paulis[0])]
        for a in range(4):
            for b in range(4):
                if a == b == 0:
                    continue
                ops.append(math.sqrt(p / 15) * xp.kron(paulis[a], paulis[b]))
    else:
        raise ValueError("depolarizing supports 1 or 2 qubits")
    return KrausChannel("depolarizing", ops, {"p": p})


def amplitude_damping(gamma: float) -> KrausChannel:
    """``|1> -> |0>`` decay with probability ``gamma`` (T1-type)."""
    g = _check_prob(gamma, "gamma")
    xp = get_backend()
    k0 = xp.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=DTYPE)
    k1 = xp.array([[0, math.sqrt(g)], [0, 0]], dtype=DTYPE)
    return KrausChannel("amplitude_damping", [k0, k1], {"gamma": g})


def phase_damping(lam: float) -> KrausChannel:
    """Pure dephasing: off-diagonals shrink by ``sqrt(1 - lam)``, populations kept."""
    lam = _check_prob(lam, "lambda")
    xp = get_backend()
    k0 = xp.array([[1, 0], [0, math.sqrt(1 - lam)]], dtype=DTYPE)
    k1 = xp.array([[0, 0], [0, math.sqrt(lam)]], dtype=DTYPE)
    return KrausChannel("phase_damping", [k0, k1], {"lam": lam})


def bit_flip(p: float) -> KrausChannel:
    p = _check_prob(p, "p")
    return KrausChannel("bit_flip", [math.sqrt(1 - p) * G.I(), math.sqrt(p) * G.X()], {"p": p})


def phase_flip(p: float) -> KrausChannel:
    p = _check_prob(p, "p")
    return KrausChannel("phase_flip", [math.sqrt(1 - p) * G.I(), math.sqrt(p) * G.Z()], {"p": p})


CHANNELS: Dict[str, Callable[..., KrausChannel]] = {
    "depolarizing": depolarizing,
    "amplitude_damping": amplitude_damping,
    "phase_damping": phase_damping,
    "bit_flip": bit_flip,
    "phase_flip": phase_flip,
}
CHANNEL_ALIASES = {
    "depolarising": "depolarizing",
    "depolarize": "depolarizing",
    "damping": "amplitude_damping",
    "dephasing": "phase_damping",
}


def make_channel(name: str, param: float, n_qubits: int = 1) -> KrausChannel:
    """Build a named channel with its error probability/rate ``param``."""
    key = CHANNEL_ALIASES.get(name.lower(), name.lower())
    if key not in CHANNELS:
        raise ValueError(f"Unknown noise channel {name!r}. Known: {sorted(CHANNELS)}")
    if key == "depolarizing":
        return depolarizing(param, n_qubits)
    if n_qubits != 1:
        raise ValueError(f"Channel {key!r} acts on a single qubit")
    return CHANNELS[key](param)


# --------------------------------------------------------------------------
# Trajectory sampling
# --------------------------------------------------------------------------
def sample_index(probs: Sequence[float], rng: Any) -> int:
    """Draw an index from a (possibly unnormalized) discrete distribution."""
    total = float(sum(probs))
    u = draw_uniform(rng) * total
    cum = 0.0
    last_positive = 0
    for i, p in enumerate(probs):
        if p > 0.0:
            last_positive = i
        cum += p
        if u < cum:
            return i
    return last_positive  # rounding fallback


def apply_channel_trajectory(
    mps: MPS, channel: KrausChannel, qubits: Sequence[int], rng: Any
) -> int:
    """Sample one Kraus branch and apply it to ``mps`` in place.

    Returns the index of the Kraus operator that fired. Two-qubit channels
    require adjacent qubits (the compiler inserts SWAPs otherwise). The state
    stays normalized; non-unitary branches are followed by renormalization.
    """
    k = len(qubits)
    if k != channel.n_qubits:
        raise ValueError(f"Channel acts on {channel.n_qubits} qubit(s), got qubits={tuple(qubits)}")
    data = channel._sampling_data()
    xp = get_backend()

    if k == 1:
        q = qubits[0]
        if data.probs is not None:  # Pauli-type fast path: any site, no canonicalization
            idx = sample_index(data.probs, rng)
            mps.apply_1q(data.unitaries[idx], q)
            return idx
        mps.canonicalize(q)
        center = mps.tensors[q]
        cands = [xp.einsum("ts,lsr->ltr", op, center) for op in data.kraus]
        weights = [float((abs(c) ** 2).sum()) for c in cands]
        idx = sample_index(weights, rng)
        mps.tensors[q] = cands[idx] / math.sqrt(weights[idx])
        return idx

    if k == 2:
        q0, q1 = qubits
        if abs(q0 - q1) != 1:
            raise ValueError("Two-qubit channels need adjacent qubits; route with SWAPs first")
        flip = q0 > q1
        i = min(q0, q1)
        if data.probs is not None:
            idx = sample_index(data.probs, rng)
            mps.apply_2q(data.unitaries[idx], q0, q1)
            return idx
        mps.prepare_two_site(i)
        theta = mps.two_site_tensor(i)
        cands, weights = [], []
        for op in data.kraus:
            op4 = reorder_two_qubit_matrix(op) if flip else op
            c = xp.einsum("abst,lstr->labr", op4.reshape(2, 2, 2, 2), theta)
            cands.append(c)
            weights.append(float((abs(c) ** 2).sum()))
        idx = sample_index(weights, rng)
        mps.set_two_site_tensor(i, cands[idx] / math.sqrt(weights[idx]))
        return idx

    raise ValueError("Only 1- and 2-qubit channels are supported")
