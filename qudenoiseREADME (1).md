# QuDenoise

A from-scratch, pure-Python **Matrix Product State (MPS) quantum circuit simulator** with
**Kraus-channel noise (quantum trajectories)** and a **quantum-autoencoder (QAE) denoiser**.
Runs on NumPy everywhere; optionally runs every tensor operation on a GPU through CuPy.

## Install

```bash
pip install qudenoise            # NumPy/SciPy only (CPU)
pip install "qudenoise[viz]"     # + matplotlib for plotting examples
```

### GPU (optional)

CuPy publishes one wheel per CUDA version, so QuDenoise does **not** pin a CuPy variant. Install the one
matching your toolkit yourself, e.g.

```bash
pip install cupy-cuda12x         # CUDA 12.x
pip install cupy-cuda11x         # CUDA 11.x
```

(`pip install "qudenoise[gpu]"` pulls the generic `cupy` source package, which needs a local CUDA toolchain.)
CuPy is imported lazily: without it QuDenoise imports and runs with no errors or warnings. With
`device="auto"` (default) the GPU is used when available and this is logged once
(`QuDenoise: running on GPU (CuPy)` / `running on CPU (NumPy)`); force a backend with
`Simulator(device="cupy")` or `device="numpy"`.

## Quick start

```python
from qudenoise import Circuit, Simulator

c = Circuit(20).h(0)
for i in range(19):
    c.cnot(i, i + 1)

sim = Simulator(bond_dim=8)              # chi cap; also truncation_threshold=...
state = sim.run(c)                       # -> MPS
print(state.bond_dimensions(), state.fidelity_estimate)
samples, report = sim.sample(c, shots=1000, seed=1)
```

Non-adjacent two-qubit gates are routed with SWAP chains automatically; every insertion is logged
(logger `qudenoise`, INFO) and recorded in `sim.last_report.routing_log`.

### Noise

```python
c = Circuit(2).h(0).cnot(0, 1).depolarizing(0, 0.05).amplitude_damping(1, 0.1)
res = Simulator().run_observable(c, lambda m: observables.expectation(m, {0: gates.Z(), 1: gates.Z()}).real,
                                 n_trajectories=2000, seed=0)
print(res.mean, "+/-", res.sem)
```

Channels: `depolarizing` (1q/2q), `amplitude_damping`, `phase_damping`, `bit_flip`, `phase_flip`, or any
`KrausChannel`. Trajectories are seeded per index (`spawn_seeds`), so results are identical for any worker
count. CPU runs with 8+ trajectories use a process pool; GPU runs are sequential in-process.

### Quantum autoencoder

```python
from qudenoise import QAE
qae = QAE(n_qubits=4, n_latent_qubits=2, ansatz_depth=2, seed=1)
qae.fit(training_states, epochs=60, lr=0.1)      # MPS / Circuit / dense vectors
clean_estimate = qae.denoise(noisy_state)
```

Cost = 1 - mean probability of the trash qubits being `|0..0>`, computed directly from the MPS. Gradients
use the parameter-shift rule through `qudenoise.qae.compute_gradient`; pass `gradient_fn=` to plug in a JAX
(`[autodiff]`) backend without changing the API.

### CLI

```bash
qudenoise run circuit.json --qubits 20 --bond-dim 32 --shots 1000 [--device auto|numpy|cupy] [-o out.json]
qudenoise train-qae config.yaml
```

Circuit JSON: `{"n_qubits": N, "ops": [{"gate": "h", "qubits": [0]}, {"gate": "rz", "qubits": [1], "params": [0.3]},
{"noise": "depolarizing", "qubits": [0], "param": 0.01}]}`. A `.py` file defining `circuit` or
`build_circuit(n)` also works (executed as ordinary Python - only run files you trust). See
`examples/qae_config.yaml`.

## Conventions and limits

* Big-endian qubit order (`|q0 q1 ...>`); rotations are `exp(-i theta P / 2)`.
* `MPS.to_dense()` refuses above 24 qubits unless forced. `qudenoise.reference` (dense/density-matrix
  simulators) exists to validate the MPS code in tests; it is not a supported product feature.
* With truncation, `state.truncation_error`, `state.fidelity_estimate` (product of `1 - eps_k`) and
  `state.fidelity_lower_bound` track accuracy loss.

## Development

```bash
pip install -e ".[test]" && pytest
```

Tests run on NumPy and, when CuPy + a GPU are present, are repeated on CuPy (backend-parity check).
See `docs/design_notes.md`.

MIT licensed.
