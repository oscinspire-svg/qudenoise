"""Expectation values, overlaps, fidelity, reduced density matrices, sampling.

All routines are transfer-matrix contractions over the MPS (``O(n chi^3)``)
and therefore work for any qubit count, unlike dense validation code.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from . import gates as G
from .backend import DTYPE, get_backend, to_numpy
from .mps import MPS


def overlap(a: MPS, b: MPS) -> complex:
    """``<a|b>`` (not normalized)."""
    if a.n_qubits != b.n_qubits:
        raise ValueError("MPS sizes differ")
    xp = get_backend()
    env = xp.ones((1, 1), dtype=DTYPE)
    for ta, tb in zip(a.tensors, b.tensors):
        env = xp.einsum("xy,xsz,ysw->zw", env, ta.conj(), tb)
    return complex(env[0, 0])


def norm(mps: MPS) -> float:
    """Norm from a full contraction (independent of the canonical bookkeeping)."""
    return math.sqrt(max(overlap(mps, mps).real, 0.0))


def fidelity(a: MPS, b: MPS) -> float:
    """``|<a|b>|^2 / (<a|a><b|b>)``."""
    na, nb = overlap(a, a).real, overlap(b, b).real
    return float(abs(overlap(a, b)) ** 2 / (na * nb))


def expectation(mps: MPS, operators: Mapping[int, Any], normalize: bool = True) -> complex:
    """Expectation value of a product of single-qubit operators.

    ``operators`` maps qubit index -> ``2x2`` matrix (identity elsewhere).
    Returns ``<psi| (x)_q O_q |psi>`` divided by ``<psi|psi>`` if ``normalize``.
    """
    xp = get_backend()
    for q in operators:
        if not 0 <= q < mps.n_qubits:
            raise ValueError(f"qubit {q} out of range")
    env = xp.ones((1, 1), dtype=DTYPE)
    for i, t in enumerate(mps.tensors):
        ket = xp.einsum("ts,lsr->ltr", operators[i], t) if i in operators else t
        env = xp.einsum("xy,xsz,ysw->zw", env, t.conj(), ket)
    value = complex(env[0, 0])
    if normalize:
        value /= overlap(mps, mps).real
    return value


def probability_of_zero(mps: MPS, qubits: Iterable[int]) -> float:
    """Probability that all ``qubits`` are measured in ``|0>``."""
    xp = get_backend()
    proj = xp.array([[1, 0], [0, 0]], dtype=DTYPE)
    return float(expectation(mps, {q: proj for q in qubits}).real)


def bloch_vector(mps: MPS, qubit: int) -> Any:
    """``(<X>, <Y>, <Z>)`` of one qubit as a host array."""
    xp = get_backend()
    vals = [expectation(mps, {qubit: op()}).real for op in (G.X, G.Y, G.Z)]
    return to_numpy(xp.array(vals))


def reduced_density_matrix(mps: MPS, qubits: Sequence[int], normalize: bool = True) -> Any:
    """Reduced density matrix of ``qubits`` (sorted ascending, big-endian).

    Cost is ``O(n chi^3 + 4^k chi^2)`` for ``k`` kept qubits: intended for small
    subsystems.
    """
    xp = get_backend()
    keep = sorted(set(int(q) for q in qubits))
    if not keep or keep[0] < 0 or keep[-1] >= mps.n_qubits:
        raise ValueError("invalid qubit list")
    env = xp.ones((1, 1, 1, 1), dtype=DTYPE)  # [ket-open, bra-open, ket-bond, bra-bond]
    for i, t in enumerate(mps.tensors):
        if i in keep:
            k, b = env.shape[0], env.shape[1]
            new = xp.einsum("kbxy,xsz,ytw->ksbtzw", env, t, t.conj())
            env = new.reshape(k * 2, b * 2, new.shape[4], new.shape[5])
        else:
            env = xp.einsum("kbxy,xsz,ysw->kbzw", env, t, t.conj())
    rho = env[:, :, 0, 0]
    if normalize:
        rho = rho / xp.trace(rho).real
    return rho


def entanglement_entropy(mps: MPS, bond: int) -> float:
    """Von Neumann entropy (bits) across the cut between sites ``bond`` and ``bond+1``."""
    if not 0 <= bond < mps.n_qubits - 1:
        raise ValueError("bond out of range")
    xp = get_backend()
    work = mps.copy().canonicalize(bond)
    t = work.tensors[bond]
    dl, _, dr = t.shape
    s = to_numpy(xp.linalg.svd(t.reshape(dl * 2, dr), compute_uv=False))
    p = s**2
    p = p / p.sum()
    return float(-sum(float(x) * math.log2(float(x)) for x in p if x > 1e-300))


def sample_bitstrings(
    mps: MPS, shots: int, rng: Optional[Any] = None, seed: Optional[int] = None
) -> Any:
    """Draw ``shots`` computational-basis samples exactly from the MPS.

    Uses the right-canonical sweep (perfect sampling, no Markov chain).
    Returns a host ``int`` array of shape ``(shots, n_qubits)`` with qubit 0 in
    column 0.
    """
    from .backend import get_rng

    xp = get_backend()
    if rng is None:
        rng = get_rng(seed)
    work = mps.copy().canonicalize(0)
    vec = xp.ones((shots, 1), dtype=DTYPE)
    rows = xp.arange(shots)
    columns = []
    for t in work.tensors:
        m = xp.einsum("bl,lsr->bsr", vec, t)
        probs = (abs(m) ** 2).sum(axis=2)  # (shots, 2)
        total = probs.sum(axis=1)
        p1 = probs[:, 1] / xp.maximum(total, 1e-300)
        u = rng.random(shots)
        s = (u < p1).astype("int64")
        chosen = m[rows, s, :]
        vec = chosen / xp.sqrt(xp.maximum(probs[rows, s], 1e-300))[:, None]
        columns.append(s)
    return to_numpy(xp.stack(columns, axis=1))


def counts_from_samples(samples: Any) -> Dict[str, int]:
    """Histogram of bitstrings (qubit 0 = leftmost character)."""
    counts: Dict[str, int] = {}
    for row in samples.tolist():
        key = "".join(str(b) for b in row)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
