#!/usr/bin/env python3
"""
reliability_toolkit.py — Extensive All-in-One Reliability & RAMS Program

A single-file Python toolkit for practical reliability engineering, RAMS, and
basic FRACAS analytics. Designed for engineers who want powerful functionality
without a heavy setup. Works on Windows/macOS/Linux.

What's new in this revision
---------------------------
• Prevents the "SystemExit: 0" message in notebooks/REPLs by avoiding a hard
  process exit in interactive environments (keeps standard exit codes for real CLI).
• Adds a built-in self-test command (`selftest`) with deterministic unit tests.
• Minor CLI robustness: command handlers return exit codes; `main` propagates them.

Key features
------------
1) Distributions: Weibull (2-param), Exponential, Lognormal.
2) Estimation: MLE (SciPy if available; otherwise robust grid/coordinate search).
3) Censored data: right-censoring (suspension) support.
4) Metrics: MTBF/MTTF, reliability R(t), unreliability F(t), hazard h(t), Bx life,
   confidence intervals (bootstrap helper), lambda-hat, eta/beta, etc.
5) RAMS: series/parallel system reliability, availability (A), maintainability snippets,
   spare parts demand via Poisson approximation.
6) FRACAS analytics: ingest failure logs (CSV/Excel), Pareto chart by part # or failure mode,
   time-between-failures (TBF) dashboard, rolling failure rate per 1k operating hours.
7) Reporting: export clean Excel report with summary, parameters, and charts (openpyxl/xlsxwriter).
8) CLI: analyze a dataset in one line; interactive GUI available via Tkinter.

Install deps
------------
python -m pip install numpy pandas matplotlib openpyxl
# Optional (recommended for faster, more accurate MLE and CIs):
python -m pip install scipy xlsxwriter

Usage
-----
CLI examples:
  python reliability_toolkit.py fit --dist weibull --csv failures.csv --time-col Hours --failed-col Failed --censor 0
  python reliability_toolkit.py pareto --csv fracas.csv --category-col PartNumber --count-col Failures
  python reliability_toolkit.py report --csv failures.csv --time-col Hours --failed-col Failed --dist weibull --out report.xlsx
  python reliability_toolkit.py system --config system.json  # see SystemConfig example below
  python reliability_toolkit.py gui
  python reliability_toolkit.py selftest  # run deterministic built-in tests

Data expectations
-----------------
- For life data: a table with at least a time column (operating time to failure or
  suspension) and a binary column Failed (1=failure, 0=censored/suspended).
- For Pareto: a table with categorical column and a count column OR raw rows to be counted.

License: MIT. No warranty. Use professional judgment.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try SciPy (recommended). Fallback to numpy-only methods if missing.
try:
    from scipy import optimize, stats
    SCIPY_AVAILABLE = True
except Exception:
    optimize = None  # type: ignore
    stats = None     # type: ignore
    SCIPY_AVAILABLE = False

__all__ = [
    "FitResult", "Distribution", "Weibull", "Exponential", "Lognormal",
    "fit_distribution", "summarize_fit", "series_reliability", "parallel_reliability",
    "availability", "expected_spares_poisson", "export_report", "build_parser", "main"
]

# ----------------------------- Utility helpers ----------------------------- #

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF for plotting."""
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _in_interactive_session() -> bool:
    """Best-effort detection of notebooks/REPLs where sys.exit is undesirable."""
    try:
        if hasattr(sys, "ps1"):
            return True  # Python REPL
        # IPython/Jupyter
        try:
            from IPython import get_ipython  # type: ignore
            if get_ipython() is not None:
                return True
        except Exception:
            pass
    except Exception:
        pass
    # Common test/IDE env markers
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if os.environ.get("PYCHARM_HOSTED") == "1":
        return True
    if os.environ.get("PYTHONINSPECT") == "1":
        return True
    return False


def _safe_exit(code: int) -> None:
    """Exit the process in CLI contexts; no-op in interactive sessions/tests."""
    try:
        if _in_interactive_session():
            # Don't hard-exit; surface the return code if nonzero
            if code != 0:
                print(f"(Non-interactive exit suppressed) Return code: {code}")
            return
    except Exception:
        pass
    sys.exit(code)


# ----------------------------- Data ingestion ------------------------------ #

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path)
    elif ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ------------------------- Distribution primitives ------------------------- #
@dataclass
class FitResult:
    dist: str
    params: Dict[str, float]
    loglik: float
    n: int
    failures: int
    censored: int
    cov: Optional[np.ndarray] = None  # covariance of MLEs if available

    def param(self, name: str, default: float = float("nan")) -> float:
        return self.params.get(name, default)


class Distribution:
    name = "base"

    def nll(self, t: np.ndarray, failed: np.ndarray) -> float:
        raise NotImplementedError

    def fit(self, t: np.ndarray, failed: np.ndarray) -> FitResult:
        raise NotImplementedError

    def R(self, t: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    def h(self, t: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    def B(self, p: float, **params) -> float:
        """Return Bp life: time by which fraction p have failed (e.g., B10)."""
        raise NotImplementedError


class Weibull(Distribution):
    name = "weibull"
    # 2-parameter Weibull: shape b (beta), scale h (eta). PDF: (b/h)*(t/h)^(b-1)*exp(-(t/h)^b)

    def nll(self, t: np.ndarray, failed: np.ndarray, beta: float, eta: float) -> float:
        f = failed.astype(bool)
        t_f = t[f]
        t_s = t[~f]
        # Log-likelihood components
        ll_fail = np.sum(_safe_log(beta/eta) + (beta-1.0)*_safe_log(t_f/eta) - (t_f/eta)**beta) if t_f.size else 0.0
        ll_susp = -np.sum((t_s/eta)**beta) if t_s.size else 0.0
        return -(ll_fail + ll_susp)

    def fit(self, t: np.ndarray, failed: np.ndarray) -> FitResult:
        t = np.asarray(t, dtype=float)
        failed = np.asarray(failed, dtype=int)
        assert np.all(t > 0), "Times must be > 0"

        def nll_wrap(x):
            b, e = x
            if b <= 0 or e <= 0:
                return 1e100
            return self.nll(t, failed, b, e)

        if SCIPY_AVAILABLE:
            x0 = np.array([1.5, np.median(t)])  # heuristic start
            bounds = [(1e-6, 100.0), (1e-9, 1e12)]
            res = optimize.minimize(nll_wrap, x0, bounds=bounds, method="L-BFGS-B")
            b, e = res.x
            loglik = -nll_wrap(res.x)
            cov = None
            if res.hess_inv is not None:
                try:
                    H = res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else np.array(res.hess_inv)
                    cov = H
                except Exception:
                    cov = None
            return FitResult(self.name, {"beta": float(b), "eta": float(e)}, loglik, len(t), int(failed.sum()), int((1-failed).sum()), cov)
        else:
            # Robust grid + local search fallback
            b_grid = np.geomspace(0.5, 5.0, 60)
            e_grid = np.geomspace(np.percentile(t, 25), np.percentile(t, 90), 60)
            best = (1e100, 1.0, 1.0)
            for b in b_grid:
                for e in e_grid:
                    val = nll_wrap((b, e))
                    if val < best[0]:
                        best = (val, b, e)
            b0, e0 = best[1], best[2]
            # Simple coordinate descent
            for _ in range(40):
                # refine eta
                e_candidates = np.geomspace(e0*0.6, e0*1.6, 25)
                vals = [nll_wrap((b0, ec)) for ec in e_candidates]
                e0 = e_candidates[int(np.argmin(vals))]
                # refine beta
                b_candidates = np.geomspace(max(b0*0.6, 1e-3), b0*1.6, 25)
                vals = [nll_wrap((bc, e0)) for bc in b_candidates]
                b0 = b_candidates[int(np.argmin(vals))]
            loglik = -nll_wrap((b0, e0))
            return FitResult(self.name, {"beta": float(b0), "eta": float(e0)}, loglik, len(t), int(failed.sum()), int((1-failed).sum()), None)

    def R(self, t: np.ndarray, beta: float, eta: float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-(t/eta)**beta)

    def h(self, t: np.ndarray, beta: float, eta: float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (beta/eta) * (t/eta)**(beta-1.0)

    def B(self, p: float, beta: float, eta: float) -> float:
        # p is failure fraction (e.g., 0.1 for B10 life)
        return eta * (-np.log(1.0 - p))**(1.0/beta)


class Exponential(Distribution):
    name = "exponential"

    def nll(self, t: np.ndarray, failed: np.ndarray, lam: float) -> float:
        f = failed.astype(bool)
        t_f = t[f]
        t_s = t[~f]
        ll_fail = np.sum(_safe_log(lam) - lam*t_f) if t_f.size else 0.0
        ll_susp = -lam*np.sum(t_s) if t_s.size else 0.0
        return -(ll_fail + ll_susp)

    def fit(self, t: np.ndarray, failed: np.ndarray) -> FitResult:
        t = np.asarray(t, dtype=float)
        failed = np.asarray(failed, dtype=int)
        assert np.all(t > 0), "Times must be > 0"

        # Closed-form MLE for exponential with right-censoring: lambda = failures / total_time
        total_time = np.sum(t)
        r = failed.sum()
        lam = r / total_time if total_time > 0 else float("nan")
        return FitResult(self.name, {"lambda": float(lam), "MTBF": float(1.0/lam) if lam > 0 else float("inf")}, -self.nll(t, failed, lam), len(t), int(r), int((1-failed).sum()), None)

    def R(self, t: np.ndarray, lam: float, **_) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-lam*t)

    def h(self, t: np.ndarray, lam: float, **_) -> np.ndarray:
        return np.full_like(np.asarray(t, dtype=float), lam)

    def B(self, p: float, lam: float, **_) -> float:
        return -np.log(1.0-p)/lam


class Lognormal(Distribution):
    name = "lognormal"

    def nll(self, t: np.ndarray, failed: np.ndarray, mu: float, sigma: float) -> float:
        f = failed.astype(bool)
        t_f = t[f]
        t_s = t[~f]
        ll_fail = -np.sum(_safe_log(t_f) + 0.5*np.log(2*np.pi*sigma**2) + (np.log(t_f)-mu)**2/(2*sigma**2)) if t_f.size else 0.0
        # Survival for suspensions
        if SCIPY_AVAILABLE:
            sf = stats.lognorm.sf(t_s, s=sigma, scale=np.exp(mu)) if t_s.size else np.array([])
        else:
            # Approximate survival using error function
            z = (np.log(np.clip(t_s, 1e-12, None)) - mu) / (sigma*math.sqrt(2)) if t_s.size else np.array([])
            from math import erf
            sf = 0.5 - 0.5*erf(z)
        ll_susp = np.sum(np.log(np.clip(sf, 1e-300, None))) if t_s.size else 0.0
        return -(ll_fail + ll_susp)

    def fit(self, t: np.ndarray, failed: np.ndarray) -> FitResult:
        t = np.asarray(t, dtype=float)
        failed = np.asarray(failed, dtype=int)
        assert np.all(t > 0), "Times must be > 0"
        # Start from log of failed items only if any; otherwise all
        base = np.log(t[failed.astype(bool)]) if failed.any() else np.log(t)
        mu0, s0 = float(np.mean(base)), float(np.std(base) + 1e-6)

        def nll_wrap(x):
            mu, s = x
            if s <= 0:
                return 1e100
            return self.nll(t, failed, mu, s)

        if SCIPY_AVAILABLE:
            res = optimize.minimize(nll_wrap, np.array([mu0, s0]), method="L-BFGS-B", bounds=[(None, None), (1e-6, 10.0)])
            mu, s = res.x
            return FitResult(self.name, {"mu": float(mu), "sigma": float(s)}, -nll_wrap(res.x), len(t), int(failed.sum()), int((1-failed).sum()), None)
        else:
            # Simple grid + coordinate descent
            mu, s = mu0, s0
            for _ in range(40):
                mu_grid = np.linspace(mu-1.0, mu+1.0, 25)
                vals = [nll_wrap((mg, s)) for mg in mu_grid]
                mu = mu_grid[int(np.argmin(vals))]
                s_grid = np.linspace(max(s*0.6, 1e-3), s*1.6, 25)
                vals = [nll_wrap((mu, sg)) for sg in s_grid]
                s = s_grid[int(np.argmin(vals))]
            return FitResult(self.name, {"mu": float(mu), "sigma": float(s)}, -nll_wrap((mu, s)), len(t), int(failed.sum()), int((1-failed).sum()), None)


DIST_MAP: Dict[str, Distribution] = {
    "weibull": Weibull(),
    "exp": Exponential(),
    "exponential": Exponential(),
    "lognormal": Lognormal(),
}

# --------------------------- Confidence intervals -------------------------- #

def bootstrap_ci(func, t: np.ndarray, failed: np.ndarray, reps: int = 1000, alpha: float = 0.05,
                 random_state: Optional[int] = 42) -> Tuple[float, float, float]:
    """Generic bootstrap for a scalar metric function->float."""
    rng = np.random.default_rng(random_state)
    n = len(t)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        vals.append(func(t[idx], failed[idx]))
    vals = np.sort(np.array(vals))
    lo = np.quantile(vals, alpha/2)
    hi = np.quantile(vals, 1 - alpha/2)
    return float(np.mean(vals)), float(lo), float(hi)


# ------------------------------- RAMS tools -------------------------------- #

def series_reliability(Rs: List[float]) -> float:
    return float(np.prod(Rs))


def parallel_reliability(Rs: List[float]) -> float:
    return float(1 - np.prod([1-r for r in Rs]))


def availability(MTBF: float, MTTR: float, MTTF: Optional[float] = None, mission_time: Optional[float] = None) -> float:
    """Steady-state availability approximation: A = MTBF/(MTBF+MTTR)."""
    if MTBF <= 0:
        return 0.0
    return MTBF/(MTBF + max(MTTR, 1e-12))


def expected_spares_poisson(failure_rate_per_hour: float, hours: float, service_level: float = 0.95) -> int:
    """Return the k such that P[X<=k] >= service_level for X~Poisson(lam)."""
    lam = max(failure_rate_per_hour, 0.0) * max(hours, 0.0)
    if SCIPY_AVAILABLE:
        return int(math.ceil(stats.poisson.ppf(service_level, lam)))
    # fallback brute force
    k, cdf = 0, math.exp(-lam)
    prob = cdf
    while cdf < service_level and k < 10_000:
        k += 1
        prob *= lam/k
        cdf += prob
    return k


# ----------------------------- Plotting helpers ---------------------------- #

def plot_weibull_probability(ax, t: np.ndarray, failed: np.ndarray, fit: FitResult):
    """Weibull probability plot: ln(-ln(1-F)) vs ln(t)."""
    f = failed.astype(bool)
    x = t[f]
    if x.size == 0:
        ax.text(0.5, 0.5, "No failures to plot", ha="center")
        return
    xs, ys = ecdf(x)
    F = ys
    y = np.log(-np.log(1 - F))
    xlog = np.log(xs)
    ax.scatter(xlog, y, label="Empirical")
    # Fit line: y = beta*ln(t) - beta*ln(eta)
    b, e = fit.param("beta"), fit.param("eta")
    xline = np.linspace(xlog.min()*0.95, xlog.max()*1.05, 200)
    yline = b * xline - b * np.log(e)
    ax.plot(xline, yline, label=f"Weibull fit β={b:.3g}, η={e:.3g}")
    ax.set_xlabel("ln(time)")
    ax.set_ylabel("ln(-ln(1-F))")
    ax.legend()
    ax.grid(True, which="both", ls=":")


def plot_hazard(ax, dist: Distribution, fit: FitResult, tmax: float):
    t = np.linspace(1e-6, tmax, 300)
    h = dist.h(t, **fit.params)
    ax.plot(t, h)
    ax.set_xlabel("Time")
    ax.set_ylabel("Hazard rate h(t)")
    ax.grid(True, ls=":")


def pareto_chart(ax, labels: List[str], counts: List[int]):
    order = np.argsort(-np.array(counts))
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    cum = np.cumsum(counts) / max(sum(counts), 1)
    ax.bar(labels, counts)
    ax2 = ax.twinx()
    ax2.plot(range(len(labels)), cum, marker="o")
    ax2.set_ylabel("Cumulative %")
    ax.set_ylabel("Count")
    ax.set_title("Pareto of Failures")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", ls=":")


# ------------------------------- Core engine ------------------------------- #

def fit_distribution(dist_name: str, t: np.ndarray, failed: np.ndarray) -> FitResult:
    if dist_name.lower() not in DIST_MAP:
        raise ValueError(f"Unknown distribution: {dist_name}. Options: {list(DIST_MAP)})")
    dist = DIST_MAP[dist_name.lower()]
    return dist.fit(t, failed)


def summarize_fit(f: FitResult, timescale: str = "hours") -> pd.DataFrame:
    rows = [{"Metric": "Distribution", "Value": f.dist},
            {"Metric": "Failures", "Value": f.failures},
            {"Metric": "Censored", "Value": f.censored},
            {"Metric": "LogLikelihood", "Value": f.loglik}]
    for k, v in f.params.items():
        rows.append({"Metric": k, "Value": v})
    # Add common derived metrics
    if f.dist == "weibull":
        beta, eta = f.param("beta"), f.param("eta")
        mtbf = eta * math.gamma(1 + 1/beta)
        rows.append({"Metric": f"MTBF ({timescale})", "Value": mtbf})
        rows.append({"Metric": "B10 life", "Value": Weibull().B(0.10, beta=beta, eta=eta)})
        rows.append({"Metric": "B50 (median)", "Value": Weibull().B(0.50, beta=beta, eta=eta)})
    elif f.dist == "exponential":
        lam = f.param("lambda")
        rows.append({"Metric": f"MTBF ({timescale})", "Value": 1.0/lam if lam>0 else float("inf")})
        rows.append({"Metric": "B10 life", "Value": Exponential().B(0.10, lam=lam)})
        rows.append({"Metric": "B50 (median)", "Value": Exponential().B(0.50, lam=lam)})
    elif f.dist == "lognormal":
        mu, s = f.param("mu"), f.param("sigma")
        # mean of lognormal
        rows.append({"Metric": f"MTTF ({timescale})", "Value": math.exp(mu + 0.5*s*s)})
        rows.append({"Metric": "B10 life", "Value": Lognormal().B(0.10, mu=mu, sigma=s)})
        rows.append({"Metric": "B50 (median)", "Value": math.exp(mu)})
    return pd.DataFrame(rows)


def export_report(out_path: str, fit: FitResult, t: np.ndarray, failed: np.ndarray,
                  pareto: Optional[Tuple[List[str], List[int]]] = None,
                  meta: Optional[Dict[str, Any]] = None) -> None:
    """Create an Excel report with summary sheet and charts.
       Requires openpyxl/xlsxwriter; falls back to CSVs if Excel writer not present."""
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            summary = summarize_fit(fit)
            summary.to_excel(writer, index=False, sheet_name="Summary")

            df = pd.DataFrame({"Time": t, "Failed": failed})
            df.to_excel(writer, index=False, sheet_name="Data")

            # Create charts as images via matplotlib, then insert
            charts_dir = os.path.join(os.path.dirname(out_path) or ".", "_charts_tmp")
            os.makedirs(charts_dir, exist_ok=True)

            # Weibull prob plot (if applicable) or hazard plot otherwise
            fig, ax = plt.subplots(figsize=(6,4))
            dist = DIST_MAP[fit.dist]
            if fit.dist == "weibull":
                plot_weibull_probability(ax, t, failed, fit)
                fig.tight_layout()
                p1 = os.path.join(charts_dir, "weibull_prob.png")
                fig.savefig(p1, dpi=160)
            else:
                plot_hazard(ax, dist, fit, tmax=float(np.percentile(t, 95)))
                fig.tight_layout()
                p1 = os.path.join(charts_dir, "hazard.png")
                fig.savefig(p1, dpi=160)
            plt.close(fig)

            # Pareto if provided
            p2 = None
            if pareto is not None:
                labels, counts = pareto
                fig2, ax2 = plt.subplots(figsize=(6,4))
                pareto_chart(ax2, labels, counts)
                fig2.tight_layout()
                p2 = os.path.join(charts_dir, "pareto.png")
                fig2.savefig(p2, dpi=160)
                plt.close(fig2)

            ws = writer.sheets["Summary"]
            # Insert images
            ws.insert_image(1, 4, p1, {"x_scale": 0.9, "y_scale": 0.9})
            if p2:
                ws.insert_image(22, 4, p2, {"x_scale": 0.9, "y_scale": 0.9})

            # Metadata
            if meta:
                meta_df = pd.DataFrame([meta])
                meta_df.to_excel(writer, index=False, sheet_name="Meta")

    except Exception as e:
        # Fallback: write CSVs side-by-side
        base = os.path.splitext(out_path)[0]
        summarize_fit(fit).to_csv(base + "_summary.csv", index=False)
        pd.DataFrame({"Time": t, "Failed": failed}).to_csv(base + "_data.csv", index=False)
        sys.stderr.write(f"Excel writer not available or failed: {e}\nWrote CSV fallbacks next to output.\n")


# ------------------------------- CLI actions ------------------------------- #

def action_fit(args) -> int:
    df = load_table(args.csv)
    t = df[args.time_col].to_numpy(dtype=float)
    if args.failed_col in df.columns:
        failed = df[args.failed_col].astype(int).to_numpy()
    else:
        # If no failed_col, assume all are failures unless a censor flag is specified
        failed = np.ones_like(t, dtype=int)
        if args.censor is not None and args.censor_col in df.columns:
            failed = (df[args.censor_col] != args.censor).astype(int).to_numpy()
    fit = fit_distribution(args.dist, t, failed)
    print(summarize_fit(fit).to_string(index=False))

    if args.plot:
        fig, ax = plt.subplots(figsize=(7,5))
        dist = DIST_MAP[fit.dist]
        if fit.dist == "weibull":
            plot_weibull_probability(ax, t, failed, fit)
        else:
            plot_hazard(ax, dist, fit, tmax=float(np.percentile(t, 95)))
        fig.tight_layout()
        plt.show()
    return 0


def action_pareto(args) -> int:
    df = load_table(args.csv)
    if args.count_col and args.count_col in df.columns:
        counts = df[args.count_col].groupby(df[args.category_col]).sum().sort_values(ascending=False)
    else:
        counts = df[args.category_col].value_counts()
    labels = counts.index.astype(str).tolist()
    nums = counts.values.astype(int).tolist()
    out = pd.DataFrame({"Category": labels, "Count": nums})
    print(out.to_string(index=False))

    if args.plot:
        fig, ax = plt.subplots(figsize=(8,5))
        pareto_chart(ax, labels, nums)
        fig.tight_layout()
        plt.show()
    return 0


def action_report(args) -> int:
    df = load_table(args.csv)
    t = df[args.time_col].to_numpy(dtype=float)
    if args.failed_col in df.columns:
        failed = df[args.failed_col].astype(int).to_numpy()
    else:
        failed = np.ones_like(t, dtype=int)
        if args.censor is not None and args.censor_col in df.columns:
            failed = (df[args.censor_col] != args.censor).astype(int).to_numpy()

    fit = fit_distribution(args.dist, t, failed)

    pareto = None
    if args.pareto_category_col:
        if args.pareto_count_col and args.pareto_count_col in df.columns:
            counts = df[args.pareto_count_col].groupby(df[args.pareto_category_col]).sum().sort_values(ascending=False)
        else:
            counts = df[args.pareto_category_col].value_counts()
        pareto = (counts.index.astype(str).tolist(), counts.values.astype(int).tolist())

    meta = {
        "source_file": os.path.abspath(args.csv),
        "distribution": args.dist,
        "time_col": args.time_col,
        "failed_col": args.failed_col,
    }
    export_report(args.out, fit, t, failed, pareto=pareto, meta=meta)
    print(f"Report written to: {args.out}")
    return 0


# ------------------------------ System modeling ---------------------------- #
@dataclass
class ComponentModel:
    name: str
    dist: str
    params: Dict[str, float]

    def R(self, t: float) -> float:
        d = DIST_MAP[self.dist]
        return float(np.clip(d.R(np.array([t]), **self.params)[0], 0.0, 1.0))


def load_system_config(path: str) -> Dict[str, Any]:
    """JSON schema example:
    {
      "time": 1000,  # mission time in hours
      "blocks": [
         {"type": "series", "name": "Chain A", "items": [
             {"type": "component", "name": "PSU", "dist": "exponential", "params": {"lambda": 1/5000}},
             {"type": "parallel", "name": "Redundant Fans", "items": [
                  {"type": "component", "name": "Fan1", "dist": "weibull", "params": {"beta": 1.8, "eta": 8000}},
                  {"type": "component", "name": "Fan2", "dist": "weibull", "params": {"beta": 1.8, "eta": 8000}}
             ]}
         ]}
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_block(block: Dict[str, Any], t: float) -> float:
    btype = block.get("type")
    if btype == "component":
        cm = ComponentModel(block["name"], block["dist"], block["params"])
        return cm.R(t)
    elif btype == "series":
        Rs = [eval_block(it, t) for it in block.get("items", [])]
        return series_reliability(Rs)
    elif btype == "parallel":
        Rs = [eval_block(it, t) for it in block.get("items", [])]
        return parallel_reliability(Rs)
    else:
        raise ValueError(f"Unknown block type: {btype}")


def action_system(args) -> int:
    config = load_system_config(args.config)
    t = float(config.get("time", 1000.0))
    blocks = config.get("blocks", [])
    R_blocks = [eval_block(b, t) for b in blocks]
    if len(R_blocks) == 1:
        R_sys = R_blocks[0]
    else:
        # Top-level series of blocks unless specified otherwise
        R_sys = series_reliability(R_blocks)
    print(json.dumps({"mission_time": t, "R_system": R_sys, "unreliability": 1-R_sys}, indent=2))
    return 0


# --------------------------------- GUI ------------------------------------- #

def action_gui(_args=None) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk, messagebox
    except Exception as e:
        print("Tkinter not available:", e)
        return 1

    root = tk.Tk()
    root.title("Reliability Toolkit")
    root.geometry("820x600")

    # Variables
    csv_path = tk.StringVar()
    time_col = tk.StringVar(value="Time")
    failed_col = tk.StringVar(value="Failed")
    dist_name = tk.StringVar(value="weibull")

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    def browse():
        p = filedialog.askopenfilename(title="Select CSV/Excel", filetypes=[("Data", "*.csv *.xlsx *.xlsm *.xls")])
        if p:
            csv_path.set(p)
            try:
                df = load_table(p)
                cols = list(df.columns)
                time_col.set(cols[0] if cols else "Time")
                if "Failed" in cols:
                    failed_col.set("Failed")
            except Exception as ex:
                messagebox.showerror("Error", str(ex))

    def run_fit():
        try:
            df = load_table(csv_path.get())
            t = df[time_col.get()].to_numpy(dtype=float)
            f = df[failed_col.get()].astype(int).to_numpy()
            fit = fit_distribution(dist_name.get(), t, f)
            out = summarize_fit(fit).to_string(index=False)
            text.delete("1.0", tk.END)
            text.insert(tk.END, out)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # Controls
    ttk.Button(frame, text="Browse Data", command=browse).grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=csv_path, width=70).grid(row=0, column=1, columnspan=3, sticky="we")
    ttk.Label(frame, text="Time column").grid(row=1, column=0, sticky="e")
    ttk.Entry(frame, textvariable=time_col, width=20).grid(row=1, column=1, sticky="w")
    ttk.Label(frame, text="Failed column").grid(row=1, column=2, sticky="e")
    ttk.Entry(frame, textvariable=failed_col, width=20).grid(row=1, column=3, sticky="w")

    ttk.Label(frame, text="Distribution").grid(row=2, column=0, sticky="e")
    ttk.Combobox(frame, textvariable=dist_name, values=list(DIST_MAP.keys())).grid(row=2, column=1, sticky="w")

    ttk.Button(frame, text="Fit", command=run_fit).grid(row=2, column=3, sticky="e")

    text = tk.Text(frame, height=25)
    text.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=10)

    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(3, weight=1)

    root.mainloop()
    return 0


# ------------------------------- Self Tests -------------------------------- #

def _almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def action_selftest(_args=None) -> int:
    """Run a set of deterministic unit tests. Returns 0 on success, >0 on failure."""
    failures: List[str] = []

    # Test 1: Exponential reliability math
    lam = 0.001
    r1000 = Exponential().R(np.array([1000.0]), lam=lam)[0]
    if not _almost_equal(r1000, math.exp(-1.0), 1e-9):
        failures.append("Exponential R(t) mismatch at t=1000, lam=0.001")

    # Test 2: Weibull reduces to exponential at beta=1 for B(p)
    eta = 1000.0
    b10_weib = Weibull().B(0.1, beta=1.0, eta=eta)
    b10_exp = Exponential().B(0.1, lam=1.0/eta)
    if not _almost_equal(b10_weib, b10_exp, 1e-9):
        failures.append("Weibull B10 != Exponential B10 at beta=1")

    # Test 3: Series/Parallel combinators
    if not _almost_equal(series_reliability([0.9, 0.8]), 0.72):
        failures.append("Series reliability incorrect")
    if not _almost_equal(parallel_reliability([0.9, 0.8]), 0.98):
        failures.append("Parallel reliability incorrect")

    # Test 4: Poisson spares monotonicity + zero lambda edge case
    s0 = expected_spares_poisson(0.0, 1000.0, 0.95)
    if s0 != 0:
        failures.append("Poisson spares at lam=0 should be 0")
    s95 = expected_spares_poisson(0.002, 100.0, 0.95)
    s99 = expected_spares_poisson(0.002, 100.0, 0.99)
    if s99 < s95:
        failures.append("Poisson spares not non-decreasing with service level")

    # Test 5: Exponential MLE with simple hand data (no censoring)
    t = np.array([10.0, 10.0, 10.0])
    f = np.array([1, 1, 1])
    fit_exp = Exponential().fit(t, f)
    if not _almost_equal(fit_exp.param("lambda"), 3.0/30.0):
        failures.append("Exponential MLE lambda incorrect for simple dataset")

    # Test 6: Lognormal median equals exp(mu)
    mu, sigma = math.log(100.0), 0.5
    med = Lognormal().B(0.5, mu=mu, sigma=sigma)
    if abs(med - math.exp(mu))/math.exp(mu) > 1e-6:
        failures.append("Lognormal median mismatch")

    if failures:
        print("Self-test failures (" + str(len(failures)) + "):")
        for msg in failures:
            print(" -", msg)
        return 1
    print("All self-tests passed.")
    return 0


# --------------------------------- Parser ---------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reliability & RAMS Toolkit (single-file)")
    sub = p.add_subparsers(dest="cmd")

    # fit
    sp = sub.add_parser("fit", help="Fit a life distribution to failure/censored data")
    sp.add_argument("--csv", required=True, help="CSV/Excel file with data")
    sp.add_argument("--time-col", required=True, help="Column containing time-to-event")
    sp.add_argument("--failed-col", required=False, default="Failed", help="Column with 1=failure,0=censored")
    sp.add_argument("--censor-col", default="Failed", help="Column to interpret with --censor if no failed-col")
    sp.add_argument("--censor", type=int, default=None, help="Value indicating censored in --censor-col if using alt schema")
    sp.add_argument("--dist", choices=list(DIST_MAP.keys()), default="weibull")
    sp.add_argument("--plot", action="store_true", help="Show probability/hazard plot")
    sp.set_defaults(func=action_fit)

    # pareto
    sp2 = sub.add_parser("pareto", help="Pareto chart from FRACAS log")
    sp2.add_argument("--csv", required=True)
    sp2.add_argument("--category-col", required=True)
    sp2.add_argument("--count-col", required=False, default=None, help="If omitted, counts occurrences of category")
    sp2.add_argument("--plot", action="store_true")
    sp2.set_defaults(func=action_pareto)

    # report
    sp3 = sub.add_parser("report", help="End-to-end fit + Excel report with charts")
    sp3.add_argument("--csv", required=True)
    sp3.add_argument("--time-col", required=True)
    sp3.add_argument("--failed-col", required=False, default="Failed")
    sp3.add_argument("--dist", choices=list(DIST_MAP.keys()), default="weibull")
    sp3.add_argument("--out", required=True)
    sp3.add_argument("--pareto-category-col", default=None)
    sp3.add_argument("--pareto-count-col", default=None)
    sp3.set_defaults(func=action_report)

    # system modeling
    sp4 = sub.add_parser("system", help="Evaluate system reliability from JSON config")
    sp4.add_argument("--config", required=True, help="Path to system configuration JSON")
    sp4.set_defaults(func=action_system)

    # GUI
    sp5 = sub.add_parser("gui", help="Launch a minimal GUI")
    sp5.set_defaults(func=action_gui)

    # self tests
    sp6 = sub.add_parser("selftest", help="Run built-in deterministic tests")
    sp6.set_defaults(func=action_selftest)

    return p


# --------------------------------- Main ------------------------------------ #

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        rc = args.func(args)
        if isinstance(rc, int):
            return rc
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130
    except Exception as e:
        print("Error:", e)
        return 1


if __name__ == "__main__":
    _safe_exit(main())
