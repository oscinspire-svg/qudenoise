# QuDenoise

**A pure-Python Matrix Product State (MPS) quantum circuit simulator with Kraus-channel noise
(quantum trajectories) and a quantum-autoencoder (QAE) denoiser.**

- **MPS simulation** with hard bond-dimension caps and/or adaptive truncation, and tracked truncation error
- **Noise** via Kraus channels unravelled into quantum trajectories, with seeded, reproducible, parallel averaging
- **Observables**: single-qubit and multi-qubit correlators, reduced density matrices, entanglement entropy, exact bitstring sampling
- **QAE denoiser**: trainable encoder/decoder that projects noisy states back onto a learned subspace
- **NumPy on CPU, CuPy on GPU**, selected at runtime. CuPy is optional and never required
- **CLI** for sampling circuits and training QAEs from config files

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Conventions](#3-conventions)
4. [Building circuits](#4-building-circuits)
5. [Running simulations](#5-running-simulations)
6. [The MPS object](#6-the-mps-object)
7. [Observables and measurement](#7-observables-and-measurement)
8. [Noise](#8-noise)
9. [Quantum autoencoder](#9-quantum-autoencoder-qae)
10. [Command-line interface](#10-command-line-interface)
11. [Backends and GPU](#11-backends-and-gpu)
12. [Accuracy and truncation](#12-accuracy-and-truncation)
13. [Supported features and limits](#13-supported-features-and-limits)
14. [Examples](#14-examples)
15. [Development](#15-development)
16. [License](#16-license)

---

## 1. Installation

Requires **Python 3.9+**. Core dependencies: `numpy>=1.22`, `scipy>=1.8`, `PyYAML>=5.4`.

```bash
pip install qudenoise                 # CPU (NumPy)
pip install "qudenoise[viz]"          # + matplotlib
pip install "qudenoise[autodiff]"     # + jax (for custom gradient_fn)
pip install "qudenoise[test]"         # + pytest
```

### GPU (optional)

CuPy ships one wheel per CUDA version, so QuDenoise does not pin one. Install the wheel
matching your toolkit:

```bash
pip install cupy-cuda12x    # CUDA 12.x
pip install cupy-cuda11x    # CUDA 11.x
```

`pip install "qudenoise[gpu]"` pulls the generic `cupy` source package, which needs a local CUDA
toolchain. Without CuPy, QuDenoise imports and runs with no errors or warnings. See
[Backends and GPU](#11-backends-and-gpu).

---

## 2. Quick start

### Noiseless circuit

```python
from qudenoise import Circuit, Simulator

c = Circuit(20).h(0)
for i in range(19):
    c.cnot(i, i + 1)

sim = Simulator(bond_dim=8)                  # chi cap
state = sim.run(c)                           # -> MPS
print(state.bond_dimensions())               # [2, 2, ..., 2]
print(state.fidelity_estimate)               # 1.0 (no truncation occurred)

samples, report = sim.sample(c, shots=1000, seed=1)   # (1000, 20) int array
```

### Correlated observable

```python
from qudenoise import gates, observables as obs

zz = obs.expectation(state, {0: gates.Z(), 19: gates.Z()})   # <Z_0 Z_19>
```

### Noisy circuit

```python
from qudenoise import Circuit, Simulator, gates, observables as obs

c = Circuit(2).h(0).cnot(0, 1).depolarizing(0, 0.05).amplitude_damping(1, 0.1)

def zz(mps):
    return obs.expectation(mps, {0: gates.Z(), 1: gates.Z()}).real

res = Simulator().run_observable(c, zz, n_trajectories=2000, seed=0)
print(res.mean, "+/-", res.sem)
```

### Quantum autoencoder

```python
from qudenoise import QAE

qae = QAE(n_qubits=4, n_latent_qubits=2, ansatz_depth=2, seed=1)
qae.fit(training_states, epochs=60, lr=0.1)        # MPS / Circuit / dense vectors
clean_estimate = qae.denoise(noisy_state)
```

---

## 3. Conventions

| Topic | Convention |
|---|---|
| Qubit order | **Big-endian**: qubit 0 is the leftmost MPS site and the most significant bit of the dense index. Bitstrings read `q0 q1 q2 ...` |
| Two-qubit gate basis | `\|q0 q1>` = `\|00>, \|01>, \|10>, \|11>`; `cnot(c, t)` has `c` as control |
| Rotations | `exp(-i θ P / 2)` for Pauli string `P` (compatible with parameter-shift) |
| Precision | `complex128` everywhere |
| MPS tensors | shape `(D_left, 2, D_right)`, boundary bonds of size 1 |
| Randomness | No global RNG. Seeds are passed explicitly; trajectory `i` uses child `i` of `SeedSequence(seed)` |
| Logging | Logger name `qudenoise`; INFO reports device choice and every SWAP-routing insertion |

---

## 4. Building circuits

`Circuit(n_qubits)` is a lightweight, backend-independent instruction list. All builder methods
return `self`, so calls chain.

```python
c = Circuit(4).h(0).cnot(0, 1).rz(2, 0.3).rzz(1, 2, 0.5).depolarizing(0, 0.01)
```

### Gate library

| Gate | Method | Qubits | Params |
|---|---|---|---|
| Identity, Pauli | `i, x, y, z` | 1 | none |
| Hadamard | `h` | 1 | none |
| Phase family | `s, sdg, t, tdg` | 1 | none |
| Rotations | `rx, ry, rz` | 1 | `theta` |
| Phase | `p` | 1 | `phi` (`diag(1, e^{iφ})`) |
| CNOT | `cnot` (alias `cx`) | 2 | none |
| CZ, SWAP | `cz, swap` | 2 | none |
| Pauli-pair rotations | `rxx, ryy, rzz` | 2 | `theta` |
| Generic entangler | `entangler(a, b, ax, ay, az)` | 2 | `exp(-i/2 (ax·XX + ay·YY + az·ZZ))` |

Name aliases for `add_gate`: `cx → cnot`, `id → i`, `phase → p`.

### Other construction methods

```python
c.add_gate("rzz", [0, 3], 0.7)              # by name
c.unitary(matrix, [0, 1])                   # arbitrary 2x2 or 4x4 unitary
c.add_noise_channel("depolarizing", [0], 0.01)   # by name, or pass a KrausChannel
c.extend(other_circuit)                     # append (same n_qubits)
c.copy()
c.inverse()                                 # adjoint; raises if noise present
len(c); c.instructions; c.has_noise
```

### Non-adjacent two-qubit operations

MPS gates act on neighbouring sites. A two-qubit gate (or two-qubit noise channel) on
non-adjacent qubits is **automatically wrapped in SWAP chains** (out and back). This is never
silent:

- each insertion is logged at INFO on the `qudenoise` logger
- `circuit.routing_log` / `circuit.n_swaps_inserted` / `simulator.last_report.routing_log`
  record them

Long-range gates are costly in an MPS. Where you can, order qubits so interacting pairs are
close.

### Compile step

`circuit.compile(route=True)` returns a flat list of `Op(op_type, qubits, data, name)` records
(`op_type` is `"gate"` or `"noise"`). The simulator does this for you.

### Serialization

```python
spec = c.to_dict()                  # {"n_qubits": N, "ops": [...]}
c2 = Circuit.from_dict(spec)
```

Limits: custom `unitary` gates and multi-parameter channels are not JSON-serializable. See
the [JSON circuit format](#json-circuit-format).

---

## 5. Running simulations

### `Simulator`

```python
Simulator(
    device="auto",                  # "auto" | "numpy"/"cpu" | "cupy"/"cuda"/"gpu"
    bond_dim=None,                  # hard cap chi on every bond
    truncation_threshold=None,      # adaptive: drop normalized singular values below this
    parallel=None,                  # None=auto, True/False to force (True ignored on GPU)
    max_workers=None,               # process count (default: CPU count)
)
```

| Method | Returns | Purpose |
|---|---|---|
| `initial_state(n_qubits, init_state=None)` | `MPS` | Fresh product state (e.g. `"0101"`) with this simulator's truncation policy |
| `run(circuit, mps=None, n_trajectories=1, seed=None)` | `MPS` if `n_trajectories == 1`, else `list[MPS]` | Execute and return final state(s). Input `mps` is not modified |
| `run_observable(circuit, observable, n_trajectories=1, seed=None, mps=None)` | `ObservableResult` | Trajectory-average `observable(mps)` |
| `sample(circuit, shots, seed=None, mps=None)` | `(samples, SimulationReport)` | Computational-basis bitstrings, shape `(shots, n_qubits)` |
| `last_report` | `SimulationReport` | Summary of the most recent call |

**Noiseless circuits** run once regardless of `n_trajectories` (all trajectories are identical).

**`sample()`** draws exact samples from one state when the circuit is noiseless. For noisy
circuits it runs **one independent trajectory per shot** and takes one sample from each,
which is the statistically exact way to sample a channel.

### `ObservableResult`

| Field | Meaning |
|---|---|
| `mean` | Trajectory average (scalar or array) |
| `sem` | Standard error of the mean (zeros for one trajectory) |
| `values` | Per-trajectory values, shape `(n_trajectories, ...)` |
| `report` | `SimulationReport` |

The observable function can return a float or an array (for example, a vector of per-qubit ⟨Z⟩ values).

> **Parallel workers:** the observable is evaluated inside worker processes. It must be a
> **module-level function** (or `functools.partial` of one) on platforms that start workers by
> `spawn` (Windows, macOS). Lambdas work with `fork` (Linux).

### `SimulationReport`

`device`, `n_qubits`, `n_ops`, `n_noise_ops`, `swaps_inserted`, `n_trajectories`, `parallel`,
`truncation_errors`, `fidelity_estimates`, `fidelity_lower_bounds`, `max_bond_dims` (all
per-trajectory lists), `routing_log`, and the property `max_truncation_error`.

### Parallelism and reproducibility

- **CPU:** with `parallel=None`, runs of **8 or more** trajectories use a process pool.
- **GPU:** trajectories run sequentially on the device (the `parallel` flag is ignored).
- Results are **identical for any worker count** because trajectory `i` always gets the same
  seed for a given master `seed`.

---

## 6. The MPS object

```python
from qudenoise import MPS

MPS(n_qubits, bond_dim=None, truncation_threshold=None,
    init_state=None, device="auto", svd_cutoff=1e-14)

MPS.from_dense(vector, bond_dim=None, truncation_threshold=None, normalize=True)
MPS.from_tensors(tensors, bond_dim=None, truncation_threshold=None)
```

| Member | Description |
|---|---|
| `tensors`, `n_qubits`, `center` | Site tensors `(Dl, 2, Dr)` and the orthogonality centre |
| `bond_dimensions()`, `max_bond_dimension` | Internal bond sizes |
| `norm()`, `normalize()` | Cheap thanks to the canonical centre |
| `canonicalize(site)` | Move the orthogonality centre |
| `apply_gate(matrix, qubits)`, `apply_1q`, `apply_2q` | Low-level gate application (`apply_2q` needs adjacent qubits; the `Circuit`/`Simulator` path routes for you) |
| `copy()` | Deep copy including truncation bookkeeping |
| `to_dense(force=False)` | Dense statevector (validation only; see limits below) |
| `project_and_drop_trailing(n_keep)` | Project trailing qubits onto `\|0>` and remove them; returns `(mps, probability)` |
| `extend_with_zeros(n_extra)` | Append `\|0>` qubits on the right |
| `to_host_tensors()` | Tensors as NumPy arrays |
| `truncation_error`, `n_truncations`, `truncation_history` | Truncation bookkeeping |
| `fidelity_estimate`, `fidelity_lower_bound`, `reset_truncation_error()` | See [Accuracy](#12-accuracy-and-truncation) |

`to_dense()` warns above 20 qubits and **refuses above 24** unless `force=True`.

---

## 7. Observables and measurement

All functions live in `qudenoise.observables` and take an `MPS`.

### Expectation values and correlators

```python
from qudenoise import gates as G, observables as obs

obs.expectation(mps, {0: G.Z()})                       # single-qubit  <Z0>
obs.expectation(mps, {0: G.Z(), 5: G.Z()})             # two-qubit     <Z0 Z5>
obs.expectation(mps, {0: G.X(), 2: G.Y(), 7: G.Z()})   # N-qubit       <X0 Y2 Z7>
```

`expectation(mps, operators, normalize=True)` computes `⟨ψ| ⊗_q O_q |ψ⟩ / ⟨ψ|ψ⟩`, where
`operators` maps **qubit index → 2×2 matrix** and every other qubit gets the identity.

- Works for **any number of qubits, adjacent or not**, in one contraction pass
- Accepts any 2×2 matrix (Paulis, projectors, custom operators), not only Paulis
- Returns `complex`; take `.real` for Hermitian operators
- A **sum** of terms (for example `ZZ + XX`) needs one call per term
- For a non-product operator on a small subset, use `reduced_density_matrix`

**Pauli-string helper** (copy-paste):

```python
from qudenoise import gates as G, observables as obs

_P = {"X": G.X, "Y": G.Y, "Z": G.Z}

def pauli_string(mps, string, qubits):
    """<P_0(q_0) P_1(q_1) ...> for e.g. string='ZXY', qubits=[0, 2, 5]."""
    return obs.expectation(mps, {q: _P[p]() for p, q in zip(string.upper(), qubits)}).real
```

Use it inside `run_observable` via `functools.partial`.

### Other functions

| Function | Returns |
|---|---|
| `overlap(a, b)` | `⟨a\|b⟩` |
| `norm(mps)` | State norm |
| `fidelity(a, b)` | `\|⟨a\|b⟩\|² / (⟨a\|a⟩⟨b\|b⟩)` |
| `probability_of_zero(mps, qubits)` | Probability all listed qubits read `0` |
| `bloch_vector(mps, qubit)` | `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)` as a host array |
| `reduced_density_matrix(mps, qubits, normalize=True)` | `2^k × 2^k` matrix of the kept qubits (sorted ascending). Cost `O(n χ³ + 4^k χ²)`, so use for small subsystems |
| `entanglement_entropy(mps, bond)` | Von Neumann entropy in **bits** across the cut between sites `bond` and `bond+1` |
| `sample_bitstrings(mps, shots, rng=None, seed=None)` | Exact (perfect) samples, `(shots, n_qubits)` int array |
| `counts_from_samples(samples)` | `{"0101": 12, ...}` sorted by key |

---

## 8. Noise

Noise is simulated with **quantum trajectories**: at each noisy location a Kraus operator
`K_k` is drawn with probability `‖K_k|ψ⟩‖²` and the state is replaced by `K_k|ψ⟩/‖·‖`.
Averaging an observable over many trajectories reproduces the density-matrix result.

### Built-in channels

| Channel | Builder method | Qubits | Parameter |
|---|---|---|---|
| Depolarizing | `c.depolarizing(qubits, p)` | 1 or 2 | error probability `p` |
| Amplitude damping | `c.amplitude_damping(q, gamma)` | 1 | decay probability `γ` (`\|1> → \|0>`) |
| Phase damping | `c.phase_damping(q, lam)` | 1 | dephasing `λ` |
| Bit flip | `c.bit_flip(q, p)` | 1 | `p` |
| Phase flip | `c.phase_flip(q, p)` | 1 | `p` |

Name aliases for `add_noise_channel`: `depolarising`/`depolarize → depolarizing`,
`damping → amplitude_damping`, `dephasing → phase_damping`.

Single-qubit depolarizing shrinks the Bloch vector by `1 − 4p/3`. Two-qubit depolarizing
(`c.depolarizing([a, b], p)`) uses the 15 non-identity two-qubit Paulis and shrinks
correlators by `1 − 16p/15`.

### Custom channels

```python
from qudenoise import KrausChannel

ch = KrausChannel("my_channel", [K0, K1, ...])   # square 2^k x 2^k Kraus operators
c.add_noise_channel(ch, [0])
```

Operators are validated for trace preservation (`‖ΣK†K − I‖ ≤ 1e-10`). Only **1- and 2-qubit**
channels are supported. Useful members: `completeness_error()`, `kraus_ops()`,
`is_scaled_unitary`.

### Performance notes

- **Pauli-type channels** (depolarizing, bit flip, phase flip) take a state-independent fast
  path: a Kraus operator is drawn up front and applied as a rescaled unitary, with no
  canonicalization for single-qubit channels.
- Non-unitary channels (damping) use local norms after moving the orthogonality centre.
- Unbiased averages need many trajectories for small effects. Watch `ObservableResult.sem`.

---

## 9. Quantum autoencoder (QAE)

A parameterized unitary `U(θ)` acts on an n-qubit state. Qubits `0 … n_latent−1` are the
**latent** register and the rest are the **trash** register. Training drives the trash qubits
to `|0…0⟩` for the training states:

```
C(θ) = 1 − mean_i ⟨ψ_i| U† (I_latent ⊗ |0..0⟩⟨0..0|_trash) U |ψ_i⟩
```

The overlap is computed directly from the MPS (no swap test).

```python
from qudenoise import QAE

qae = QAE(
    n_qubits=4, n_latent_qubits=2,
    ansatz_depth=2,                 # rotation + entangler blocks
    bond_dim=None, truncation_threshold=None,
    gradient_fn=None,               # optional (cost_fn, params) -> list[float]
    seed=1,
)
```

Constraints: `n_qubits >= 2` and `1 <= n_latent_qubits < n_qubits`.

### Ansatz

`ansatz_depth` blocks of `[RY, RZ on every qubit] → [brick of generic two-qubit entanglers on
neighbouring pairs]`, then a final `[RY, RZ]` layer. All gates are on adjacent qubits (no
routing), and every parameter enters through one `exp(−iθP/2)` gate, so the two-term
parameter-shift rule is exact for untruncated MPS. The parameter count is `qae.n_params`.

### Training

```python
result = qae.fit(
    training_states,        # MPS objects, state-prep Circuits, or dense vectors
    epochs=100, lr=0.1,     # Adam
    batch_size=None,        # random subset per epoch
    tol=None,               # stop early once full-set cost < tol
    seed=None,
    restarts=1,             # independent random inits; best final cost wins
    gradient="parameter_shift",     # or "finite_difference"
    callback=None,          # callback(epoch, cost, params)
)
```

`FitResult` has `params`, `cost_history`, `initial_cost`, `final_cost`, `epochs_run`,
`cost_evaluations`, `converged`. Cost per epoch is roughly `2 × n_params × n_states` MPS
simulations.

### Using the trained model

| Method | Description |
|---|---|
| `encode(state, return_probability=False)` | Apply `U`, project trash onto `\|0..0>`, return normalized latent `MPS`. With `return_probability=True` also returns the surviving weight (low means the input lies outside the learned subspace) |
| `decode(latent)` | Append `\|0>` trash qubits, apply `U†` |
| `denoise(state)` | `decode(encode(state))` |
| `denoise_fidelity(noisy, clean)` | `{"before", "after", "gain"}` fidelities to `clean` |
| `cost(states, params=None)`, `trash_probability(state, params=None)` | Evaluate the objective |
| `circuit(params=None)` | The encoder as a `Circuit` |
| `to_dict()` / `QAE.from_dict(spec)` | Save/restore model (config + trained parameters) |

`denoise_fidelity` can report a **negative gain** if the "noisy" state is already inside the
learned subspace. That is expected.

### Gradients

All gradient computation goes through `qudenoise.qae.compute_gradient(cost_fn, params,
method="parameter_shift" | "finite_difference")`. Pass `gradient_fn=` to `QAE` to plug in
another backend (for example JAX) without changing the rest of the API. With MPS truncation
enabled, parameter-shift gradients are only approximate.

---

## 10. Command-line interface

```bash
qudenoise --version
qudenoise [-v] run circuit.json [options]
qudenoise [-v] train-qae config.yaml
```

`-v/--verbose` goes **before** the subcommand and enables INFO logging (device, routing, progress).
Errors print `qudenoise: error: ...` and exit with code 2.

### `qudenoise run`

Simulates a circuit file and samples bitstrings.

| Option | Default | Description |
|---|---|---|
| `circuit` | required | `.json` file, or `.py` defining `circuit` or `build_circuit(n_qubits)` |
| `-n, --qubits` | from file | Number of qubits (overrides/validates the file; required with `build_circuit`) |
| `-d, --bond-dim` | unbounded | Maximum bond dimension |
| `--threshold` | none | Truncation-weight threshold per SVD |
| `-k, --shots` | 1000 | Number of samples |
| `--seed` | none | RNG seed |
| `--device` | `auto` | `auto, cpu, numpy, cuda, gpu, cupy` |
| `--serial` | off | Disable CPU multiprocessing for noisy runs |
| `-o, --output` | stdout | Write JSON result to a file |

The output JSON contains `n_qubits`, `shots`, `device`, `bond_dim`, `counts` (most frequent
first), `swaps_inserted`, `max_bond_dim_reached`, `truncation_error`, and `fidelity_estimate`.
The CLI **samples bitstrings only**. For observables, use the Python API.

> A `.py` circuit file is executed as ordinary Python. Only run files you trust.

### JSON circuit format

```json
{
  "n_qubits": 3,
  "ops": [
    {"gate": "h", "qubits": [0]},
    {"gate": "rz", "qubits": [1], "params": [0.3]},
    {"gate": "cx", "qubits": [0, 2]},
    {"noise": "depolarizing", "qubits": [0], "param": 0.01}
  ]
}
```

For noise ops, `param` may also be written `p`, `gamma`, or `lam`.

### `qudenoise train-qae`

Trains a QAE from a YAML or JSON config and writes the model plus a report.

| Key | Required | Description |
|---|---|---|
| `n_qubits`, `n_latent_qubits` | yes | Register sizes |
| `training_states` | yes | List of circuits, each an inline circuit mapping or a file path (relative to the config) |
| `ansatz_depth` | no (2) | Ansatz depth |
| `bond_dim`, `truncation_threshold` | no | MPS truncation |
| `seed`, `device` | no | RNG seed; `auto/cpu/cuda` |
| `fit` | no | `epochs, lr, batch_size, tol, restarts, seed` |
| `test_pairs` | no | List of `{clean: <circuit>, noisy: <circuit>}` for fidelity evaluation |
| `output` | no (`qae_model.json`) | Output path, relative to the config |

The output JSON contains `model` (loadable with `QAE.from_dict`), `report` (costs, epochs,
convergence, per-pair fidelities, mean gain), and `cost_history`. See
`examples/qae_config.yaml`.

---

## 11. Backends and GPU

- The array library is **process-global state** (like matplotlib's backend). Switch with
  `qudenoise.set_backend("numpy" | "cupy")` or scoped with `with use_backend("numpy"): ...`.
- `device="auto"` (default) uses CuPy if it is importable and a CUDA device is visible, else
  NumPy. The choice is logged once.
- `device="cupy"` / `"cuda"` / `"gpu"` **raises** `RuntimeError` if unavailable.
  `"auto"` never fails.
- `gpu_available()`, `get_backend()`, `to_backend(arr)`, `to_numpy(arr)`, `get_rng(seed)` are
  exported for advanced use.
- `to_numpy` is the way to bring results back to the host.
- Set `QUDENOISE_DEBUG=1` to run an SVD self-check on backend switches (a silently wrong SVD
  would corrupt every truncation).
- SVD falls back to SciPy's more robust `gesvd` driver if the default one fails to converge.

---

## 12. Accuracy and truncation

Each truncating SVD records its discarded probability weight `ε_k`. Running totals live on every MPS:

| Property | Meaning |
|---|---|
| `truncation_error` | `Σ ε_k` |
| `fidelity_estimate` | `Π (1 − ε_k)`, the usual estimate of fidelity with the untruncated state |
| `fidelity_lower_bound` | `max(0, 1 − (Σ √ε_k)²)`, a rigorous bound |
| `truncation_history` | `[(bond_index, ε_k), ...]` |

Truncation rules (applied together): `bond_dim` is a hard cap; `truncation_threshold` drops
singular values below it (singular values normalized so their squares sum to 1); a tiny always-on
`svd_cutoff` (1e-14) removes numerical zeros. At least one singular value is always kept, and the
kept values are rescaled to preserve the norm.

**Practical advice:** increase `bond_dim` until observables stop changing, and check
`fidelity_estimate` / `report.max_truncation_error` after every run.

---

## 13. Supported features and limits

**Supported**
- 1- and 2-qubit gates (any pair, auto-routed); arbitrary 1-/2-qubit unitaries
- 1- and 2-qubit Kraus channels
- Product-of-single-qubit-operator observables on any qubit subset; reduced density matrices
- Exact computational-basis sampling
- Initial product states in the computational basis (`init_state="0101"`)

**Not supported / be aware**
- No gates on 3+ qubits (decompose them first)
- No mid-circuit measurement, reset, or classical control
- No built-in density-matrix simulation (noise is by trajectories). The `qudenoise.reference`
  dense/density-matrix simulators exist **only to validate the MPS code in tests** (around
  14 qubits for statevectors and about 6 for density matrices) and are not a supported feature
- No sums-of-operators observables, so evaluate term by term
- Long-range gates incur SWAP overhead and extra truncation
- QAE targets **small registers** (parameter-shift cost scales with parameters × states in pure
  Python) and can hit local minima. Use `restarts=`
- Custom unitaries and multi-parameter channels cannot be saved in the JSON circuit format
- CLI `run` outputs samples, not observables

---

## 14. Examples

Runnable scripts in `examples/`:

| File | Demonstrates |
|---|---|
| `ghz_state.py` | 60-qubit GHZ state at bond dimension 4, well beyond dense simulation |
| `noisy_bell_pair.py` | Depolarizing noise: trajectory average vs the exact analytic ⟨ZZ⟩ |
| `train_qae_denoiser.py` | Train a QAE on a state family and denoise perturbed states |
| `qae_config.yaml` | Config for `qudenoise train-qae` |

---

## 15. Development

```bash
pip install -e ".[test]"
pytest
```

Tests run on NumPy and, when CuPy and a GPU are present, are repeated on CuPy as a
backend-parity check. Design rationale (backend layer, canonical form, routing, trajectories,
QAE gradients, known limitations) is in `docs/design_notes.md`.

---

## 16. License

MIT.
