#!/usr/bin/env python3
"""
Benchmark: SciPy vs JAXFit fitting speed on real expressions.

Compares the current SciPy-based fitting pipeline against JAXFit
(GPU-accelerated, auto-differentiated) for the expressions used in the
RedPitaya fitting GUI.

Run
---
    cd fitting && python bench_fit.py

Outputs a table of timing results plus matplotlib comparison plots.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Ensure imports resolve ───────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_JAXFIT_DIR = str(_HERE / "jaxfit")
if _JAXFIT_DIR not in sys.path:
    sys.path.insert(0, _JAXFIT_DIR)

from scipy.optimize import curve_fit as scipy_curve_fit  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jaxfit import CurveFit  # noqa: E402

from jax_backend import (  # noqa: E402
    build_jax_model_func,
)
from model import compile_segment_expression  # noqa: E402


# =====================================================================
# Test expressions — the real models used in the lab
# =====================================================================
EXPRESSIONS: Dict[str, Tuple[str, List[str], Dict[str, float]]] = {
    "linear": (
        "m*x+c",
        ["m", "c"],
        {"m": 3.0, "c": 5.0},
    ),
    "MI_interferometer": (
        "abs(A_MI*sin(A_mod*sin(2*pi*f_mod*x+pi*phi_mod)+pi*phi_MI))**2+V_0",
        ["A_MI", "A_mod", "f_mod", "phi_mod", "phi_MI", "V_0"],
        {
            "A_MI": 2.0,
            "A_mod": 2.0,
            "f_mod": 100.0,
            "phi_mod": 0.3,
            "phi_MI": 0.1,
            "V_0": 0.5,
        },
    ),
    "damped_sine": (
        "A*exp(-gamma*x)*sin(2*pi*f*x+pi*phi)+offset",
        ["A", "gamma", "f", "phi", "offset"],
        {"A": 3.0, "gamma": 0.5, "f": 5.0, "phi": 0.2, "offset": 1.0},
    ),
    "gaussian": (
        "A*exp(-(x-mu)**2/(2*sigma**2))+bg",
        ["A", "mu", "sigma", "bg"],
        {"A": 5.0, "mu": 5.0, "sigma": 1.0, "bg": 0.2},
    ),
}

# Reasonable bounds for each expression
BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "linear": {"m": (-10, 10), "c": (-10, 20)},
    "MI_interferometer": {
        "A_MI": (0, 10),
        "A_mod": (0, 10),
        "f_mod": (0, 1e3),
        "phi_mod": (-4, 4),
        "phi_MI": (-4, 4),
        "V_0": (0, 10),
    },
    "damped_sine": {
        "A": (0, 10),
        "gamma": (0, 5),
        "f": (0, 50),
        "phi": (-4, 4),
        "offset": (-5, 5),
    },
    "gaussian": {
        "A": (0, 20),
        "mu": (0, 10),
        "sigma": (0.01, 5),
        "bg": (-5, 5),
    },
}


# =====================================================================
# SciPy model builder (mirrors model.py compile_segment_expression)
# =====================================================================
def build_scipy_model(expr: str, param_names: List[str]) -> callable:
    """Build a callable f(x, *params) compatible with scipy.curve_fit."""
    evaluator = compile_segment_expression(expr, param_names)

    def model(x, *values):
        pmap = {name: float(v) for name, v in zip(param_names, values)}
        return evaluator(np.asarray(x, dtype=float), pmap)

    return model


def build_jax_model(expr: str, param_names: List[str]) -> callable:
    """Build a callable f(x, *params) compatible with jaxfit.CurveFit."""
    func, free_names = build_jax_model_func(expr, param_names)
    return func


# =====================================================================
# Synthetic data generator
# =====================================================================
def generate_data(
    expr: str,
    param_names: List[str],
    true_params: Dict[str, float],
    n_points: int,
    x_range: Tuple[float, float] = (0, 10),
    noise_std: float = 0.1,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate noisy synthetic data from the given expression."""
    rng = np.random.default_rng(rng_seed)
    x = np.linspace(x_range[0], x_range[1], n_points)
    evaluator = compile_segment_expression(expr, param_names)
    y_clean = evaluator(x, true_params)
    y = y_clean + rng.normal(0, noise_std, size=n_points)
    return x, y


# =====================================================================
# Benchmark routines
# =====================================================================
def benchmark_single(
    name: str,
    expr: str,
    param_names: List[str],
    true_params: Dict[str, float],
    bounds_map: Dict[str, Tuple[float, float]],
    n_points: int = 100_000,
    n_repeats: int = 15,
    noise_std: float = 0.15,
) -> Dict[str, list]:
    """Benchmark scipy vs jaxfit for a single expression."""
    print(f"\n{'=' * 60}")
    print(f"  {name}:  {expr}")
    print(f"  n_points={n_points}  n_repeats={n_repeats}")
    print(f"{'=' * 60}")

    # Build bound arrays
    lo = [bounds_map[n][0] for n in param_names]
    hi = [bounds_map[n][1] for n in param_names]
    # Initial guess: midpoint of bounds (not true params, to test convergence)
    p0 = [0.5 * (lo_i + hi_i) for lo_i, hi_i in zip(lo, hi)]

    # Build models
    scipy_model = build_scipy_model(expr, param_names)
    jax_model = build_jax_model(expr, param_names)

    # Pre-warm jaxfit (first call triggers tracing)
    jcf = CurveFit(flength=n_points)
    x_warm, y_warm = generate_data(
        expr, param_names, true_params, n_points, rng_seed=0, noise_std=noise_std
    )
    print("  JAX warm-up (tracing)... ", end="", flush=True)
    warmup_t0 = time.perf_counter()
    try:
        jcf.curve_fit(jax_model, x_warm, y_warm, p0=p0, bounds=(lo, hi))
    except Exception as e:
        print(f"  [warm-up failed: {e}]")
    warmup_elapsed = time.perf_counter() - warmup_t0
    print(f"{warmup_elapsed:.2f}s")

    scipy_times = []
    jax_times = []
    scipy_residuals = []
    jax_residuals = []

    for i in range(n_repeats):
        x, y = generate_data(
            expr,
            param_names,
            true_params,
            n_points,
            rng_seed=i + 100,
            noise_std=noise_std,
        )

        # --- SciPy ---
        t0 = time.perf_counter()
        try:
            popt_sp, _ = scipy_curve_fit(
                scipy_model, x, y, p0=p0, bounds=(lo, hi), method="trf", maxfev=5000
            )
            residual_sp = float(np.sum((scipy_model(x, *popt_sp) - y) ** 2))
        except Exception:
            popt_sp = np.array(p0)
            residual_sp = float("inf")
        scipy_times.append(time.perf_counter() - t0)
        scipy_residuals.append(residual_sp)

        # --- JAXFit ---
        t0 = time.perf_counter()
        try:
            popt_jx, _ = jcf.curve_fit(jax_model, x, y, p0=p0, bounds=(lo, hi))
            y_pred_jx = np.asarray(
                jax_model(jnp.asarray(x), *[float(v) for v in popt_jx])
            )
            residual_jx = float(np.sum((y_pred_jx - y) ** 2))
        except Exception:
            popt_jx = np.array(p0)
            residual_jx = float("inf")
        jax_times.append(time.perf_counter() - t0)
        jax_residuals.append(residual_jx)

        label = f"  [{i + 1:2d}/{n_repeats}]"
        print(
            f"{label}  SciPy: {scipy_times[-1]:.4f}s (SSE={scipy_residuals[-1]:.2f})   "
            f"JAXFit: {jax_times[-1]:.4f}s (SSE={jax_residuals[-1]:.2f})"
        )

    return {
        "scipy_times": scipy_times,
        "jax_times": jax_times,
        "scipy_residuals": scipy_residuals,
        "jax_residuals": jax_residuals,
        "warmup": warmup_elapsed,
    }


# =====================================================================
# Data-size scaling benchmark
# =====================================================================
def benchmark_scaling(
    name: str,
    expr: str,
    param_names: List[str],
    true_params: Dict[str, float],
    bounds_map: Dict[str, Tuple[float, float]],
    sizes: Sequence[int] = (1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000),
    noise_std: float = 0.15,
) -> Dict[str, list]:
    """Time fitting as a function of data length."""
    print(f"\n{'=' * 60}")
    print(f"  Scaling test: {name}")
    print(f"{'=' * 60}")

    lo = [bounds_map[n][0] for n in param_names]
    hi = [bounds_map[n][1] for n in param_names]
    p0 = [0.5 * (lo_i + hi_i) for lo_i, hi_i in zip(lo, hi)]

    scipy_model = build_scipy_model(expr, param_names)
    jax_model = build_jax_model(expr, param_names)

    max_size = max(sizes)
    jcf = CurveFit(flength=max_size)
    # warm up
    x_w, y_w = generate_data(
        expr, param_names, true_params, max_size, rng_seed=0, noise_std=noise_std
    )
    try:
        jcf.curve_fit(jax_model, x_w, y_w, p0=p0, bounds=(lo, hi))
    except Exception:
        pass

    scipy_times = []
    jax_times = []
    actual_sizes = []

    for n in sizes:
        x, y = generate_data(
            expr, param_names, true_params, n, rng_seed=7, noise_std=noise_std
        )
        actual_sizes.append(n)

        t0 = time.perf_counter()
        try:
            scipy_curve_fit(
                scipy_model, x, y, p0=p0, bounds=(lo, hi), method="trf", maxfev=5000
            )
        except Exception:
            pass
        scipy_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            jcf.curve_fit(jax_model, x, y, p0=p0, bounds=(lo, hi))
        except Exception:
            pass
        jax_times.append(time.perf_counter() - t0)

        print(
            f"  n={n:>8,d}  SciPy: {scipy_times[-1]:.4f}s  JAXFit: {jax_times[-1]:.4f}s  "
            f"speedup: {scipy_times[-1] / max(jax_times[-1], 1e-9):.1f}x"
        )

    return {"sizes": actual_sizes, "scipy_times": scipy_times, "jax_times": jax_times}


# =====================================================================
# Plotting helpers
# =====================================================================
def plot_comparison(results: Dict[str, Dict], suptitle: str = "SciPy vs JAXFit"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    for ax, (name, data) in zip(axes[0], results.items()):
        iters = list(range(1, len(data["scipy_times"]) + 1))
        ax.plot(iters, data["scipy_times"], "o-", label="SciPy", color="#dc2626")
        ax.plot(iters, data["jax_times"], "s-", label="JAXFit", color="#2563eb")
        ax.set_xlabel("Fit iteration")
        ax.set_ylabel("Time (s)")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate median speedup (skip first JAX call)
        sp_med = np.median(data["scipy_times"][1:])
        jx_med = np.median(data["jax_times"][1:])
        if jx_med > 0:
            ax.text(
                0.98,
                0.95,
                f"Median speedup: {sp_med / jx_med:.1f}x",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0f2fe", alpha=0.8),
            )

    fig.tight_layout()
    return fig


def plot_scaling(scaling_data: Dict[str, Dict]):
    n = len(scaling_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle("Fit Speed vs Data Size", fontsize=14, fontweight="bold")

    for ax, (name, data) in zip(axes[0], scaling_data.items()):
        ax.plot(
            data["sizes"], data["scipy_times"], "o-", label="SciPy", color="#dc2626"
        )
        ax.plot(data["sizes"], data["jax_times"], "s-", label="JAXFit", color="#2563eb")
        ax.set_xlabel("Data points")
        ax.set_ylabel("Time (s)")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    return fig


# =====================================================================
# Summary table
# =====================================================================
def print_summary(all_results: Dict[str, Dict]):
    print("\n" + "=" * 80)
    print("  SUMMARY: median fit time (excluding first iteration)")
    print("=" * 80)
    print(
        f"  {'Expression':<25s} {'SciPy (s)':>12s} {'JAXFit (s)':>12s} {'Speedup':>10s} {'SSE match':>10s}"
    )
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")
    for name, data in all_results.items():
        sp = np.median(data["scipy_times"][1:])
        jx = np.median(data["jax_times"][1:])
        speedup = sp / jx if jx > 0 else float("inf")
        sp_sse = np.median(data["scipy_residuals"][1:])
        jx_sse = np.median(data["jax_residuals"][1:])
        match = "✓" if abs(sp_sse - jx_sse) / max(sp_sse, 1e-12) < 0.05 else "~"
        print(f"  {name:<25s} {sp:>12.4f} {jx:>12.4f} {speedup:>9.1f}x {match:>10s}")
    print("=" * 80)


# =====================================================================
# Main
# =====================================================================
def main():
    print("JAX devices:", jax.devices())
    print()

    all_results: Dict[str, Dict] = {}
    scaling_results: Dict[str, Dict] = {}

    for name, (expr, params, true_vals) in EXPRESSIONS.items():
        data = benchmark_single(
            name,
            expr,
            params,
            true_vals,
            BOUNDS[name],
            n_points=100_000,
            n_repeats=12,
        )
        all_results[name] = data

    # Scaling test on the MI expression (the real workload)
    mi_expr, mi_params, mi_true = EXPRESSIONS["MI_interferometer"]
    scaling_results["MI_interferometer"] = benchmark_scaling(
        "MI_interferometer",
        mi_expr,
        mi_params,
        mi_true,
        BOUNDS["MI_interferometer"],
        sizes=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
    )

    lin_expr, lin_params, lin_true = EXPRESSIONS["linear"]
    scaling_results["linear"] = benchmark_scaling(
        "linear",
        lin_expr,
        lin_params,
        lin_true,
        BOUNDS["linear"],
        sizes=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
    )

    print_summary(all_results)

    plot_comparison(all_results)
    plot_scaling(scaling_results)
    plt.show()


if __name__ == "__main__":
    main()
