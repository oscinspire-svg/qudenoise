"""Circuit execution on an MPS, including noisy quantum trajectories.

Layering: :mod:`qudenoise.mps` owns the low-level gate kernels (1-qubit
``einsum``; 2-qubit merge -> contract -> SVD -> truncate -> split);
:mod:`qudenoise.noise` owns Kraus sampling; this module orchestrates them:
compiling circuits (with logged SWAP routing), scheduling trajectories, and
reporting truncation error.

Parallelism
-----------
* CPU (NumPy): independent trajectories run in worker processes
  (``concurrent.futures.ProcessPoolExecutor``).
* GPU (CuPy): CUDA contexts do not fork/parallelize like NumPy processes, so
  trajectories run sequentially in-process on the device. (Batching
  trajectories along a leading tensor axis is a possible future optimization;
  see ``docs/design_notes.md``.)

Reproducibility: trajectory ``i`` receives the ``i``-th child of a
``SeedSequence(seed)``, so results do not depend on the number of workers or
on whether trajectories run in parallel.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from . import noise as N
from .backend import (
    backend_name,
    configure,
    get_backend,
    get_rng,
    set_backend,
    spawn_seeds,
    to_numpy,
)
from .circuit import Circuit, Op
from .mps import MPS
from .observables import sample_bitstrings

logger = logging.getLogger("qudenoise")

#: With ``parallel=None`` (auto), use worker processes from this many trajectories on.
PARALLEL_MIN_TRAJECTORIES = 8


@dataclass
class SimulationReport:
    """Inspectable summary of the last :meth:`Simulator.run`-style call."""

    device: str
    n_qubits: int
    n_ops: int  # compiled ops, incl. inserted SWAPs
    n_noise_ops: int
    swaps_inserted: int
    n_trajectories: int
    parallel: bool
    truncation_errors: List[float] = field(default_factory=list)  # sum of eps_k per trajectory
    fidelity_estimates: List[float] = field(default_factory=list)  # prod(1 - eps_k) per trajectory
    fidelity_lower_bounds: List[float] = field(default_factory=list)
    max_bond_dims: List[int] = field(default_factory=list)
    routing_log: List[str] = field(default_factory=list)

    @property
    def max_truncation_error(self) -> float:
        return max(self.truncation_errors) if self.truncation_errors else 0.0


@dataclass
class ObservableResult:
    """Trajectory-averaged observable."""

    mean: Any
    sem: Any  # standard error of the mean (zeros for a single trajectory)
    values: Any  # shape (n_trajectories, ...)
    report: SimulationReport


# --------------------------------------------------------------------------
# execution kernels (module-level so worker processes can import them)
# --------------------------------------------------------------------------
def execute_ops(ops: Sequence[Op], mps: MPS, rng: Any) -> MPS:
    """Apply compiled ops to ``mps`` in place, sampling one noise branch per channel."""
    for op in ops:
        if op.op_type == "gate":
            mps.apply_gate(op.data, op.qubits)
        elif op.op_type == "noise":
            N.apply_channel_trajectory(mps, op.data, op.qubits, rng)
        else:  # pragma: no cover
            raise ValueError(f"Unknown op type {op.op_type!r}")
    return mps


Task = Tuple[str, Any]  # ("state", None) | ("observable", fn) | ("samples", k)
_Stats = Tuple[float, float, float, int]  # (trunc_err, fid_est, fid_lb, max_bond)


def _run_one(ops: Sequence[Op], template: MPS, seed: int, task: Task) -> Tuple[Any, _Stats]:
    rng = get_rng(seed)
    mps = execute_ops(ops, template.copy(), rng)
    kind, arg = task
    if kind == "state":
        result: Any = mps
    elif kind == "observable":
        result = arg(mps)
    elif kind == "samples":
        result = sample_bitstrings(mps, int(arg), rng=rng)
    else:  # pragma: no cover
        raise ValueError(kind)
    stats = (
        mps.truncation_error,
        mps.fidelity_estimate,
        mps.fidelity_lower_bound,
        mps.max_bond_dimension,
    )
    return result, stats


_WORKER_CTX: dict = {}


def _init_worker(ops: Sequence[Op], template: MPS, task: Task) -> None:
    set_backend("numpy")
    _WORKER_CTX.update(ops=ops, template=template, task=task)


def _worker_run(seed: int) -> Tuple[Any, _Stats]:
    return _run_one(_WORKER_CTX["ops"], _WORKER_CTX["template"], seed, _WORKER_CTX["task"])


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
class Simulator:
    """Runs :class:`~qudenoise.circuit.Circuit` objects on an MPS.

    Parameters
    ----------
    device:
        ``"auto"`` (CuPy if available, else NumPy), ``"cpu"``/``"numpy"``, or
        ``"cuda"``/``"gpu"``/``"cupy"`` (raises if unavailable). The choice is
        logged once at construction.
    bond_dim, truncation_threshold:
        Truncation policy for MPS created by the simulator (fixed cap ``chi``
        and/or adaptive singular-value threshold). If an ``mps`` is passed to
        :meth:`run`, it keeps its own policy unless these are given here.
    parallel:
        ``None`` (default) = worker processes for ``>= 8`` trajectories on CPU;
        ``True``/``False`` force the choice (``True`` is ignored on GPU).
    max_workers:
        Process count for parallel trajectories (default: CPU count).
    """

    def __init__(
        self,
        device: str = "auto",
        bond_dim: Optional[int] = None,
        truncation_threshold: Optional[float] = None,
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.backend = configure(device)
        self.device = device
        self.bond_dim = bond_dim
        self.truncation_threshold = truncation_threshold
        self.parallel = parallel
        self.max_workers = max_workers
        self.last_report: Optional[SimulationReport] = None

    def __repr__(self) -> str:
        return (
            f"Simulator(backend={self.backend!r}, bond_dim={self.bond_dim}, "
            f"truncation_threshold={self.truncation_threshold})"
        )

    # ------------------------------------------------------------------
    def initial_state(self, n_qubits: int, init_state: Optional[str] = None) -> MPS:
        set_backend(self.backend)
        return MPS(
            n_qubits,
            bond_dim=self.bond_dim,
            truncation_threshold=self.truncation_threshold,
            init_state=init_state,
            device=self.backend,
        )

    def _prepare(self, circuit: Circuit, mps: Optional[MPS]) -> Tuple[List[Op], MPS]:
        set_backend(self.backend)  # the backend is process-global: re-assert ours
        if mps is None:
            template = self.initial_state(circuit.n_qubits)
        else:
            if mps.n_qubits != circuit.n_qubits:
                raise ValueError(
                    f"MPS has {mps.n_qubits} qubits but the circuit has {circuit.n_qubits}"
                )
            template = mps.copy()
            if self.bond_dim is not None:
                template.bond_dim = self.bond_dim
            if self.truncation_threshold is not None:
                template.truncation_threshold = self.truncation_threshold
        return circuit.compile(route=True), template

    def _use_processes(self, n_trajectories: int) -> bool:
        if n_trajectories < 2 or self.backend != "numpy":
            return False
        if (os.cpu_count() or 1) < 2 and self.max_workers is None:
            return False
        if self.parallel is None:
            return n_trajectories >= PARALLEL_MIN_TRAJECTORIES
        return bool(self.parallel)

    def _dispatch(
        self, ops: List[Op], template: MPS, seeds: List[int], task: Task
    ) -> Tuple[List[Any], List[_Stats], bool]:
        if self._use_processes(len(seeds)):
            workers = self.max_workers or min(os.cpu_count() or 1, len(seeds))
            chunk = max(1, len(seeds) // (4 * workers))
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(ops, template, task)
            ) as pool:
                out = list(pool.map(_worker_run, seeds, chunksize=chunk))
            parallel = True
        else:
            if self.parallel and self.backend != "numpy":
                logger.info("GPU backend: running trajectories sequentially on the device")
            out = [_run_one(ops, template, s, task) for s in seeds]
            parallel = False
        return [r for r, _ in out], [s for _, s in out], parallel

    def _report(
        self,
        circuit: Circuit,
        ops: List[Op],
        stats: List[_Stats],
        n_trajectories: int,
        parallel: bool,
    ) -> SimulationReport:
        report = SimulationReport(
            device=backend_name(),
            n_qubits=circuit.n_qubits,
            n_ops=len(ops),
            n_noise_ops=sum(1 for o in ops if o.op_type == "noise"),
            swaps_inserted=circuit.n_swaps_inserted,
            n_trajectories=n_trajectories,
            parallel=parallel,
            truncation_errors=[s[0] for s in stats],
            fidelity_estimates=[s[1] for s in stats],
            fidelity_lower_bounds=[s[2] for s in stats],
            max_bond_dims=[s[3] for s in stats],
            routing_log=list(circuit.routing_log),
        )
        self.last_report = report
        return report

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Circuit,
        mps: Optional[MPS] = None,
        n_trajectories: int = 1,
        seed: Optional[int] = None,
    ) -> Union[MPS, List[MPS]]:
        """Apply ``circuit`` and return the final state(s).

        Returns a single :class:`MPS` when ``n_trajectories == 1`` and a list of
        ``n_trajectories`` independent trajectory states otherwise. The input
        ``mps`` (if given) is not modified. Truncation error is available
        afterwards on each returned MPS (``.truncation_error`` etc.) and in
        :attr:`last_report`.

        Without noise every trajectory is identical, so the circuit is executed
        once and copies are returned.
        """
        if n_trajectories < 1:
            raise ValueError("n_trajectories must be >= 1")
        ops, template = self._prepare(circuit, mps)
        noisy = any(o.op_type == "noise" for o in ops)
        n_exec = n_trajectories if noisy else 1
        seeds = spawn_seeds(seed, n_exec)
        states, stats, parallel = self._dispatch(ops, template, seeds, ("state", None))
        self._report(circuit, ops, stats, n_exec, parallel)
        if n_trajectories == 1:
            return states[0]
        if not noisy:
            logger.info("Circuit has no noise: %d trajectories are identical; ran once", n_trajectories)
            return [states[0]] + [states[0].copy() for _ in range(n_trajectories - 1)]
        return states

    def run_observable(
        self,
        circuit: Circuit,
        observable: Callable[[MPS], Any],
        n_trajectories: int = 1,
        seed: Optional[int] = None,
        mps: Optional[MPS] = None,
    ) -> ObservableResult:
        """Trajectory-average ``observable(mps)`` (float or array-valued).

        The observable is evaluated inside the worker so that full states need
        not be shipped between processes. It must be a module-level function
        (or ``functools.partial`` of one) if the platform starts workers by
        ``spawn`` rather than ``fork``.
        """
        ops, template = self._prepare(circuit, mps)
        noisy = any(o.op_type == "noise" for o in ops)
        n_exec = n_trajectories if noisy else 1
        seeds = spawn_seeds(seed, n_exec)
        vals, stats, parallel = self._dispatch(ops, template, seeds, ("observable", observable))
        report = self._report(circuit, ops, stats, n_exec, parallel)
        xp = get_backend()
        arr = to_numpy(xp.asarray(vals))
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / arr.shape[0] ** 0.5 if arr.shape[0] > 1 else 0 * mean
        return ObservableResult(mean=mean, sem=sem, values=arr, report=report)

    def sample(
        self,
        circuit: Circuit,
        shots: int,
        seed: Optional[int] = None,
        mps: Optional[MPS] = None,
    ) -> Tuple[Any, SimulationReport]:
        """Draw ``shots`` computational-basis bitstrings ``(shots, n_qubits)``.

        Noiseless circuits: one exact state, ``shots`` samples from it. Noisy
        circuits: one independent trajectory per shot (the statistically exact
        way to sample a channel), one sample from each.
        """
        ops, template = self._prepare(circuit, mps)
        noisy = any(o.op_type == "noise" for o in ops)
        if noisy:
            seeds = spawn_seeds(seed, shots)
            outs, stats, parallel = self._dispatch(ops, template, seeds, ("samples", 1))
            xp = get_backend()
            samples = to_numpy(xp.concatenate([xp.asarray(o) for o in outs], axis=0))
        else:
            seeds = spawn_seeds(seed, 1)
            outs, stats, parallel = self._dispatch(ops, template, seeds, ("samples", shots))
            samples = outs[0]
        report = self._report(circuit, ops, stats, len(seeds), parallel)
        return samples, report
