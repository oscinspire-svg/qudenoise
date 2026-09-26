"""Command-line interface: a thin wrapper over the library.

    qudenoise run circuit.json --qubits 20 --bond-dim 32 --shots 1000
    qudenoise train-qae config.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__

logger = logging.getLogger("qudenoise")


class CLIError(Exception):
    """User-facing error (printed without a traceback, exit code 2)."""


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------
def _load_structured(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise CLIError(f"File not found: {path}")
    text = path.read_text()
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except Exception as exc:  # noqa: BLE001 - report any parse failure uniformly
        raise CLIError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CLIError(f"{path} must contain a mapping at the top level")
    return data


def load_circuit(path: Path, n_qubits: Optional[int] = None):
    """Load a circuit from ``.json`` (see ``Circuit.from_dict``) or ``.py``.

    A ``.py`` file must define ``circuit`` (a ``Circuit``) or
    ``build_circuit(n_qubits) -> Circuit``. It is executed as normal Python,
    so only run files you trust.
    """
    from .circuit import Circuit

    if path.suffix.lower() == ".py":
        if not path.is_file():
            raise CLIError(f"File not found: {path}")
        spec = importlib.util.spec_from_file_location("_qudenoise_user_circuit", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if hasattr(module, "build_circuit"):
            if n_qubits is None:
                raise CLIError("--qubits is required with a build_circuit(n_qubits) circuit file")
            circ = module.build_circuit(n_qubits)
        elif hasattr(module, "circuit"):
            circ = module.circuit
        else:
            raise CLIError(f"{path} must define `circuit` or `build_circuit(n_qubits)`")
        if not isinstance(circ, Circuit):
            raise CLIError("The circuit file did not produce a qudenoise Circuit")
        if n_qubits is not None and circ.n_qubits != n_qubits:
            raise CLIError(f"Circuit has {circ.n_qubits} qubits but --qubits {n_qubits} was given")
        return circ
    spec = _load_structured(path)
    try:
        circ = Circuit.from_dict(spec, n_qubits=n_qubits)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
    if n_qubits is not None and spec.get("n_qubits") not in (None, n_qubits):
        raise CLIError(f"Circuit file declares {spec['n_qubits']} qubits but --qubits {n_qubits} was given")
    return circ


def _circuit_from_entry(entry: Any, base: Path):
    """A config entry is an inline circuit dict or a path (relative to the config)."""
    from .circuit import Circuit

    if isinstance(entry, str):
        return load_circuit((base / entry) if not Path(entry).is_absolute() else Path(entry))
    if isinstance(entry, dict):
        try:
            return Circuit.from_dict(entry)
        except ValueError as exc:
            raise CLIError(str(exc)) from exc
    raise CLIError(f"Circuit entries must be a file path or a mapping, got {type(entry).__name__}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    from .observables import counts_from_samples
    from .simulator import Simulator

    circ = load_circuit(Path(args.circuit), args.qubits)
    try:
        sim = Simulator(
            device=args.device,
            bond_dim=args.bond_dim,
            truncation_threshold=args.threshold,
            parallel=False if args.serial else None,
        )
        samples, report = sim.sample(circ, args.shots, seed=args.seed)
    except (ValueError, RuntimeError) as exc:
        raise CLIError(str(exc)) from exc
    counts = counts_from_samples(samples)
    result = {
        "n_qubits": circ.n_qubits,
        "shots": args.shots,
        "device": report.device,
        "bond_dim": args.bond_dim,
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "swaps_inserted": report.swaps_inserted,
        "max_bond_dim_reached": max(report.max_bond_dims) if report.max_bond_dims else 1,
        "truncation_error": report.max_truncation_error,
        "fidelity_estimate": min(report.fidelity_estimates) if report.fidelity_estimates else 1.0,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_train_qae(args: argparse.Namespace) -> int:
    from . import observables as obs
    from .backend import backend_name, configure, get_rng
    from .mps import MPS
    from .qae import QAE
    from .simulator import execute_ops

    cfg_path = Path(args.config)
    cfg = _load_structured(cfg_path)
    base = cfg_path.parent
    for key in ("n_qubits", "n_latent_qubits", "training_states"):
        if key not in cfg:
            raise CLIError(f"Config is missing required key '{key}'")
    try:
        configure(cfg.get("device", "auto"))
        fit_cfg = cfg.get("fit", {})
        qae = QAE(
            cfg["n_qubits"],
            cfg["n_latent_qubits"],
            cfg.get("ansatz_depth", 2),
            bond_dim=cfg.get("bond_dim"),
            truncation_threshold=cfg.get("truncation_threshold"),
            seed=cfg.get("seed"),
        )

        def to_state(entry: Any) -> MPS:
            c = _circuit_from_entry(entry, base)
            return execute_ops(c.compile(), MPS(c.n_qubits, device=backend_name()), get_rng(0))

        train = [to_state(e) for e in cfg["training_states"]]
        result = qae.fit(
            train,
            epochs=fit_cfg.get("epochs", 100),
            lr=fit_cfg.get("lr", 0.1),
            batch_size=fit_cfg.get("batch_size"),
            tol=fit_cfg.get("tol"),
            restarts=fit_cfg.get("restarts", 1),
            seed=fit_cfg.get("seed", cfg.get("seed")),
        )
        report: Dict[str, Any] = {
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
            "epochs_run": result.epochs_run,
            "converged": result.converged,
        }
        pairs: List[Dict[str, float]] = []
        for pair in cfg.get("test_pairs", []):
            r = qae.denoise_fidelity(to_state(pair["noisy"]), to_state(pair["clean"]))
            pairs.append(r)
        if pairs:
            report["denoising"] = pairs
            report["mean_fidelity_gain"] = sum(p["gain"] for p in pairs) / len(pairs)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise CLIError(str(exc)) from exc
    out = {"model": qae.to_dict(), "report": report, "cost_history": result.cost_history}
    out_path = Path(cfg.get("output", "qae_model.json"))
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"QAE trained: cost {result.initial_cost:.4f} -> {result.final_cost:.4f} in {result.epochs_run} epochs")
    if "mean_fidelity_gain" in report:
        print(f"Mean denoising fidelity gain on test pairs: {report['mean_fidelity_gain']:+.4f}")
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qudenoise", description="MPS quantum circuit simulator with noise and a QAE denoiser.")
    p.add_argument("--version", action="version", version=f"qudenoise {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="log routing / device / progress information")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="simulate a circuit file and sample bitstrings")
    r.add_argument("circuit", help="circuit .json (or .py defining `circuit` / `build_circuit(n)`)")
    r.add_argument("--qubits", "-n", type=int, help="number of qubits (overrides / validates the file)")
    r.add_argument("--bond-dim", "-d", type=int, default=None, help="maximum bond dimension (default: unbounded)")
    r.add_argument("--threshold", type=float, default=None, help="truncation-weight threshold per SVD")
    r.add_argument("--shots", "-k", type=int, default=1000)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "numpy", "cuda", "gpu", "cupy"])
    r.add_argument("--serial", action="store_true", help="disable CPU multiprocessing for noisy runs")
    r.add_argument("--output", "-o", help="write the JSON result here instead of stdout")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("train-qae", help="train a quantum autoencoder from a YAML/JSON config")
    t.add_argument("config")
    t.set_defaults(func=cmd_train_qae)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .utils import setup_logging

    args = build_parser().parse_args(argv)
    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    try:
        return args.func(args)
    except CLIError as exc:
        print(f"qudenoise: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
