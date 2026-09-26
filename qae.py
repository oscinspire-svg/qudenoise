"""Quantum autoencoder (QAE) for state compression and denoising.

Model
-----
An ``n``-qubit parameterized unitary ``U(theta)`` (the *encoder*) acts on an
input state. The first ``n_latent`` qubits form the *latent* register, the
remaining ``n_trash = n - n_latent`` qubits are the *trash* register. Training
minimizes the trash cost

    C(theta) = 1 - mean_i  <psi_i| U(theta)^dag  (I_latent (x) |0..0><0..0|_trash)  U(theta) |psi_i>

i.e. one minus the average probability of finding the trash qubits in
``|0...0>`` after encoding. Because this is a classical simulator with full
state access, the overlap is computed directly from the MPS (no swap test).

* :meth:`QAE.encode` applies ``U``, projects the trash qubits onto ``|0..0>``
  and returns the normalized latent MPS on ``n_latent`` qubits.
* :meth:`QAE.decode` appends fresh ``|0>`` trash qubits and applies ``U^dag``.
* :meth:`QAE.denoise` is ``decode(encode(state))``: components of a noisy
  state that leave the learned subspace are projected away.

Ansatz
------
``depth`` blocks of ``[RY, RZ on every qubit] -> [brick of generic two-qubit
entanglers on neighbouring pairs]`` followed by a final ``[RY, RZ]`` layer.
The entangler is ``exp(-i(a XX + b YY + c ZZ)/2)`` with three parameters. All
gates act on adjacent qubits, so no SWAP routing is needed, and every
parameter enters through exactly one gate of the form ``exp(-i theta P / 2)``
with ``P`` a Pauli string, so the two-term **parameter-shift rule** is exact
(for un-truncated MPS).

Gradients
---------
All gradient computation goes through :func:`compute_gradient`. To plug in a
different gradient backend (e.g. JAX autodiff) pass ``gradient_fn`` to
:class:`QAE` - a callable ``(cost_fn, params) -> list[float]`` - without
touching anything else in the public API.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import gates as _gates
from .backend import backend_name, get_rng, to_numpy
from .circuit import Circuit
from .mps import MPS
from .observables import fidelity as _fidelity
from .observables import probability_of_zero

logger = logging.getLogger("qudenoise")

StateLike = Union[MPS, Circuit, Any]  # MPS, state-prep Circuit, or dense vector
GradientFn = Callable[[Callable[[List[float]], float], List[float]], List[float]]

# (gate name, qubits, parameter index)
_Slot = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


# ---------------------------------------------------------------------------
# gradients (the single isolation point for a future autodiff backend)
# ---------------------------------------------------------------------------
def compute_gradient(
    cost_fn: Callable[[List[float]], float],
    params: Sequence[float],
    method: str = "parameter_shift",
    shift: float = math.pi / 2,
    eps: float = 1e-6,
) -> List[float]:
    """Gradient of ``cost_fn`` at ``params``.

    ``"parameter_shift"``: ``(C(t + s) - C(t - s)) / (2 sin s)`` per parameter,
    exact for gates ``exp(-i t P / 2)`` (default ``s = pi/2``).
    ``"finite_difference"``: central differences with step ``eps``.
    """
    p = [float(v) for v in params]
    grad: List[float] = []
    if method == "parameter_shift":
        denom = 2.0 * math.sin(shift)
        for k in range(len(p)):
            plus = list(p)
            minus = list(p)
            plus[k] += shift
            minus[k] -= shift
            grad.append((cost_fn(plus) - cost_fn(minus)) / denom)
    elif method == "finite_difference":
        for k in range(len(p)):
            plus = list(p)
            minus = list(p)
            plus[k] += eps
            minus[k] -= eps
            grad.append((cost_fn(plus) - cost_fn(minus)) / (2.0 * eps))
    else:
        raise ValueError(f"Unknown gradient method {method!r}; use 'parameter_shift' or 'finite_difference'")
    return grad


class _Adam:
    """Plain Adam on Python lists of floats (parameters are tiny; host-side)."""

    def __init__(self, n: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [0.0] * n
        self.v = [0.0] * n
        self.t = 0

    def step(self, params: List[float], grad: Sequence[float]) -> List[float]:
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        out = []
        for i, (x, g) in enumerate(zip(params, grad)):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g
            out.append(x - self.lr * (self.m[i] / c1) / (math.sqrt(self.v[i] / c2) + self.eps))
        return out


@dataclass
class FitResult:
    """Outcome of :meth:`QAE.fit`."""

    params: List[float]
    cost_history: List[float] = field(default_factory=list)
    final_cost: float = float("nan")
    epochs_run: int = 0
    cost_evaluations: int = 0
    converged: bool = False

    @property
    def initial_cost(self) -> float:
        return self.cost_history[0] if self.cost_history else float("nan")


# ---------------------------------------------------------------------------
# QAE
# ---------------------------------------------------------------------------
class QAE:
    """Quantum autoencoder with an explicit trash register.

    Parameters
    ----------
    n_qubits:
        Number of qubits of the full state.
    n_latent_qubits:
        Size of the latent register (``1 <= n_latent < n_qubits``). Qubits
        ``0 .. n_latent-1`` are latent, ``n_latent .. n-1`` are trash.
    ansatz_depth:
        Number of ``rotation + entangler`` blocks.
    bond_dim, truncation_threshold:
        Optional MPS truncation applied while simulating the encoder. ``None``
        keeps whatever the input state already carries. With truncation the
        parameter-shift gradient is only approximate.
    gradient_fn:
        Optional replacement ``(cost_fn, params) -> list[float]`` for the
        default parameter-shift gradient (the hook for a future JAX backend).
    seed:
        Seed for the random parameter initialization.
    """

    def __init__(
        self,
        n_qubits: int,
        n_latent_qubits: int,
        ansatz_depth: int = 2,
        bond_dim: Optional[int] = None,
        truncation_threshold: Optional[float] = None,
        gradient_fn: Optional[GradientFn] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if not 1 <= n_latent_qubits < n_qubits:
            raise ValueError(f"n_latent_qubits must be in [1, {n_qubits - 1}]")
        if ansatz_depth < 0:
            raise ValueError("ansatz_depth must be >= 0")
        self.n_qubits = int(n_qubits)
        self.n_latent_qubits = int(n_latent_qubits)
        self.n_trash_qubits = self.n_qubits - self.n_latent_qubits
        self.ansatz_depth = int(ansatz_depth)
        self.bond_dim = bond_dim
        self.truncation_threshold = truncation_threshold
        self.gradient_fn = gradient_fn
        self._slots, self.n_params = self._build_ansatz()
        self.trash_qubits: Tuple[int, ...] = tuple(range(self.n_latent_qubits, self.n_qubits))
        self.params: List[float] = self.initial_params(seed)
        self.fit_result: Optional[FitResult] = None

    def __repr__(self) -> str:
        return (
            f"QAE(n_qubits={self.n_qubits}, n_latent_qubits={self.n_latent_qubits}, "
            f"ansatz_depth={self.ansatz_depth}, n_params={self.n_params})"
        )

    # -- ansatz --------------------------------------------------------------
    def _build_ansatz(self) -> Tuple[List[_Slot], int]:
        n = self.n_qubits
        slots: List[_Slot] = []
        k = 0

        def rotations() -> None:
            nonlocal k
            for q in range(n):
                slots.append(("ry", (q,), (k,)))
                slots.append(("rz", (q,), (k + 1,)))
                k += 2

        for d in range(self.ansatz_depth):
            rotations()
            for a in range(d % 2, n - 1, 2):
                slots.append(("entangler", (a, a + 1), (k, k + 1, k + 2)))
                k += 3
        rotations()
        return slots, k

    def initial_params(self, seed: Optional[int] = None, scale: float = 0.5) -> List[float]:
        """Random initial parameters ``N(0, scale^2)`` drawn from ``backend.get_rng``."""
        rng = get_rng(seed)
        vals = to_numpy(rng.normal(0.0, scale, self.n_params))
        return [float(v) for v in vals]

    def _resolve(self, params: Optional[Sequence[float]]) -> List[float]:
        p = self.params if params is None else [float(v) for v in params]
        if len(p) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(p)}")
        return p

    def circuit(self, params: Optional[Sequence[float]] = None) -> Circuit:
        """The encoder ``U(theta)`` as a :class:`~qudenoise.circuit.Circuit`."""
        p = self._resolve(params)
        c = Circuit(self.n_qubits)
        for name, qubits, idx in self._slots:
            c.add_gate(name, qubits, *[p[i] for i in idx])
        return c

    def _apply_unitary(self, mps: MPS, params: List[float], inverse: bool = False) -> None:
        """Apply ``U`` (or ``U^dag``) in place. All gates are on adjacent qubits."""
        slots = reversed(self._slots) if inverse else self._slots
        sign = -1.0 if inverse else 1.0
        for name, qubits, idx in slots:
            mat = _gates.get_gate(name, *[sign * params[i] for i in idx])
            mps.apply_gate(mat, qubits)

    # -- state handling ------------------------------------------------------
    def _to_mps(self, state: StateLike, n: Optional[int] = None) -> MPS:
        """Return a fresh, normalized MPS copy of ``state`` (MPS / Circuit / dense)."""
        n = self.n_qubits if n is None else n
        if isinstance(state, MPS):
            out = state.copy()
        elif isinstance(state, Circuit):
            from .simulator import execute_ops

            start = MPS(state.n_qubits, device=backend_name())
            out = execute_ops(state.compile(), start, get_rng(0))
        else:
            out = MPS.from_dense(state, self.bond_dim, self.truncation_threshold)
        if out.n_qubits != n:
            raise ValueError(f"Expected a {n}-qubit state, got {out.n_qubits} qubits")
        if self.bond_dim is not None:
            out.bond_dim = self.bond_dim
        if self.truncation_threshold is not None:
            out.truncation_threshold = self.truncation_threshold
        return out

    # -- cost ----------------------------------------------------------------
    def trash_probability(self, state: StateLike, params: Optional[Sequence[float]] = None) -> float:
        """Probability of finding all trash qubits in ``|0>`` after encoding ``state``."""
        p = self._resolve(params)
        mps = self._to_mps(state)
        self._apply_unitary(mps, p)
        return probability_of_zero(mps, self.trash_qubits)

    def cost(self, training_states: Sequence[StateLike], params: Optional[Sequence[float]] = None) -> float:
        """Trash cost ``1 - mean_i P_trash=0`` over the training states."""
        states = self._as_state_list(training_states)
        p = self._resolve(params)
        return 1.0 - sum(self.trash_probability(s, p) for s in states) / len(states)

    def _as_state_list(self, states: Union[StateLike, Sequence[StateLike]]) -> List[MPS]:
        if isinstance(states, (MPS, Circuit)):
            states = [states]
        elif hasattr(states, "ndim") and getattr(states, "ndim") == 1:
            states = [states]
        states = list(states)
        if not states:
            raise ValueError("Need at least one training state")
        # Convert once up front so repeated cost evaluations only copy MPS objects.
        return [self._to_mps(s) for s in states]

    # -- training ------------------------------------------------------------
    def fit(
        self,
        training_states: Sequence[StateLike],
        epochs: int = 100,
        lr: float = 0.1,
        batch_size: Optional[int] = None,
        tol: Optional[float] = None,
        seed: Optional[int] = None,
        restarts: int = 1,
        gradient: str = "parameter_shift",
        callback: Optional[Callable[[int, float, List[float]], None]] = None,
    ) -> FitResult:
        """Train the encoder with Adam on the trash cost.

        Parameters
        ----------
        training_states:
            MPS objects, state-preparation circuits, or dense statevectors.
        epochs, lr:
            Adam epochs and learning rate. One epoch = one gradient step.
        batch_size:
            Number of training states per gradient step (random subset each
            epoch); ``None`` uses all of them.
        tol:
            Stop early once the cost on the full training set falls below
            ``tol`` (checked each epoch when not batching).
        restarts:
            Number of independent random initializations; the best final
            cost wins. Restart 0 starts from the current ``self.params`` when
            they have not been trained yet, otherwise all restarts are random.
        gradient:
            ``"parameter_shift"`` (default) or ``"finite_difference"``; ignored
            when ``gradient_fn`` was given to the constructor.
        callback:
            ``callback(epoch, cost, params)`` called after every epoch.
        """
        if epochs < 0:
            raise ValueError("epochs must be >= 0")
        if restarts < 1:
            raise ValueError("restarts must be >= 1")
        states = self._as_state_list(training_states)
        if batch_size is not None and not 1 <= batch_size <= len(states):
            raise ValueError(f"batch_size must be in [1, {len(states)}]")

        seeds = _split_seed(seed, restarts + 1)
        rng = get_rng(seeds[-1])
        best: Optional[FitResult] = None
        for r in range(restarts):
            start = self.params if (r == 0 and self.fit_result is None) else self.initial_params(seeds[r])
            result = self._fit_once(states, list(start), epochs, lr, batch_size, tol, rng, gradient, callback)
            logger.info("QAE fit restart %d/%d: cost %.3e -> %.3e", r + 1, restarts, result.initial_cost, result.final_cost)
            if best is None or result.final_cost < best.final_cost:
                best = result
        assert best is not None
        self.params = list(best.params)
        self.fit_result = best
        return best

    def _gradient(self, cost_fn: Callable[[List[float]], float], params: List[float], method: str) -> List[float]:
        if self.gradient_fn is not None:
            return [float(g) for g in self.gradient_fn(cost_fn, params)]
        return compute_gradient(cost_fn, params, method=method)

    def _fit_once(
        self,
        states: List[MPS],
        params: List[float],
        epochs: int,
        lr: float,
        batch_size: Optional[int],
        tol: Optional[float],
        rng: Any,
        gradient: str,
        callback: Optional[Callable[[int, float, List[float]], None]],
    ) -> FitResult:
        opt = _Adam(self.n_params, lr)
        history: List[float] = []
        n_eval = 0
        converged = False
        epochs_run = 0
        for epoch in range(epochs):
            if batch_size is None or batch_size == len(states):
                batch = states
            else:
                idx = to_numpy(rng.permutation(len(states)))[:batch_size]
                batch = [states[int(i)] for i in idx]

            def cost_fn(p: List[float], _b: List[MPS] = batch) -> float:
                nonlocal n_eval
                n_eval += 1
                return 1.0 - sum(self.trash_probability(s, p) for s in _b) / len(_b)

            cost_now = cost_fn(params)
            history.append(cost_now)
            epochs_run = epoch + 1
            if callback is not None:
                callback(epoch, cost_now, params)
            if tol is not None and batch is states and cost_now < tol:
                converged = True
                break
            grad = self._gradient(cost_fn, params, gradient)
            params = opt.step(params, grad)
        final = 1.0 - sum(self.trash_probability(s, params) for s in states) / len(states)
        n_eval += len(states)
        if tol is not None and final < tol:
            converged = True
        return FitResult(list(params), history, final, epochs_run, n_eval, converged)

    # -- encode / decode / denoise --------------------------------------------
    def encode(self, state: StateLike, return_probability: bool = False) -> Any:
        """Encode ``state`` into the latent register.

        Applies ``U(theta)``, projects the trash qubits onto ``|0..0>`` and
        returns the normalized latent MPS (``n_latent_qubits`` qubits). With
        ``return_probability=True`` returns ``(latent, p_trash_zero)`` where
        the probability is the weight that survived the projection - a low
        value means the input lies outside the learned subspace. Raises
        ``ValueError`` if that weight is exactly zero.
        """
        mps = self._to_mps(state)
        self._apply_unitary(mps, self._resolve(None))
        latent, prob = mps.project_and_drop_trailing(self.n_latent_qubits)
        return (latent, prob) if return_probability else latent

    def decode(self, latent: StateLike) -> MPS:
        """Decode a latent state: append ``|0>`` trash qubits and apply ``U^dag``."""
        mps = self._to_mps(latent, n=self.n_latent_qubits).extend_with_zeros(self.n_trash_qubits)
        self._apply_unitary(mps, self._resolve(None), inverse=True)
        return mps

    def denoise(self, noisy_state: StateLike) -> MPS:
        """``decode(encode(noisy_state))``: project a state back onto the learned subspace."""
        return self.decode(self.encode(noisy_state))

    def denoise_fidelity(self, noisy_state: StateLike, clean_state: StateLike) -> Dict[str, float]:
        """Fidelity to ``clean_state`` before and after denoising (convenience for evaluation)."""
        clean = self._to_mps(clean_state)
        noisy = self._to_mps(noisy_state)
        before = _fidelity(clean, noisy)
        after = _fidelity(clean, self.denoise(noisy))
        return {"before": before, "after": after, "gain": after - before}

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_latent_qubits": self.n_latent_qubits,
            "ansatz_depth": self.ansatz_depth,
            "bond_dim": self.bond_dim,
            "truncation_threshold": self.truncation_threshold,
            "params": list(self.params),
        }

    @classmethod
    def from_dict(cls, spec: Dict[str, Any]) -> "QAE":
        qae = cls(
            spec["n_qubits"],
            spec["n_latent_qubits"],
            spec.get("ansatz_depth", 2),
            bond_dim=spec.get("bond_dim"),
            truncation_threshold=spec.get("truncation_threshold"),
        )
        if "params" in spec:
            qae.params = qae._resolve(spec["params"])
        return qae


def _split_seed(seed: Optional[int], n: int) -> List[Optional[int]]:
    """Deterministic child seeds (``None`` seed stays non-deterministic)."""
    from .backend import spawn_seeds

    return list(spawn_seeds(seed, n))
