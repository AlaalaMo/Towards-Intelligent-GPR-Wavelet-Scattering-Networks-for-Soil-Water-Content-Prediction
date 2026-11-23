from __future__ import annotations
import os
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime


# ----------------------------- Small Utilities -----------------------------

def _r2_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R² for 1-D target."""
    return r2_score(y_true, y_pred)

def _freq_axis(n_features: int, fmin: float, fmax: float) -> np.ndarray:
    """Return linear GHz axis for feature indices."""
    return np.linspace(fmin, fmax, num=n_features)

def _windows(n_features: int, size: int, stride: int) -> List[Tuple[int, int, int]]:
    """Return (start, end, center_idx) sliding windows over feature axis."""
    starts = list(range(0, max(n_features - size + 1, 1), stride))
    if not starts or starts[-1] + size < n_features:
        last = max(n_features - size, 0)
        if not starts or starts[-1] != last:
            starts.append(last)
    return [(s, min(s + size, n_features), s + min(size, n_features - s) // 2) for s in starts]

def _permute_cols_inplace(X: np.ndarray, s: int, e: int, rng: np.random.Generator) -> None:
    """Shuffle each column X[:, j] independently across samples for j in [s, e)."""
    for j in range(s, e):
        rng.shuffle(X[:, j])

def _ensure_dir(path: str) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


# ----------------------------- Core Compute -----------------------------

# CHANGED: multi-output permutation ΔR² in long format (one row per window×target)
def delta_r2_by_window_permute_multi(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    window_size: int = 128,
    stride: int = 32,
    f_min_ghz: float = 0.4,
    f_max_ghz: float = 2.0,
    n_permutations: int = 8,
    random_state: int = 42,
    depth_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute permutation ΔR² per window for each target column.

    Shuffles features inside each window across samples, re-predicts, and computes
    ΔR² = R²_baseline(target) − mean R²_permuted(target), returning long-format DataFrame.

    Args:
        model: Fitted estimator with .predict(X) → (n_samples, n_targets).
        X_test: Test features (n_samples, n_features).
        y_test: Test targets (n_samples, n_targets=4 in order [10,20,30,40] cm).
        window_size: Window length in features.
        stride: Stride between windows.
        f_min_ghz: Minimum frequency label.
        f_max_ghz: Maximum frequency label.
        n_permutations: Number of shuffles per window.
        random_state: RNG seed.
        depth_labels: Optional list of target labels; default ['10 cm','20 cm','30 cm','40 cm'].

    Returns:
        Long-format DataFrame with columns:
        ['w_start','w_end','center_idx','f_center_GHz','target_idx','target_label',
         'r2_mean','r2_std','delta_r2_mean','delta_r2_std'] and attrs['r2_baseline_per_target'].
    """
    X = np.asarray(X_test)
    y = np.asarray(y_test)
    n, nfeat = X.shape[0], X.shape[1]
    nt = y.shape[1] if y.ndim == 2 else 1
    if depth_labels is None:
        depth_labels = [f"{d} cm" for d in (10, 20, 30, 40)][:nt]

    y_base_pred = np.asarray(model.predict(X))
    r2_base_t = np.array([_r2_1d(y[:, t], y_base_pred[:, t]) for t in range(nt)])

    wins = _windows(nfeat, window_size, stride)
    faxis = _freq_axis(nfeat, f_min_ghz, f_max_ghz)

    rng_master = np.random.default_rng(random_state)
    rows = []
    for (s, e, c) in wins:
        r2_vals_t = [[] for _ in range(nt)]
        for _ in range(n_permutations):
            Xp = X.copy()
            _permute_cols_inplace(Xp, s, e, np.random.default_rng(rng_master.integers(0, 2**31 - 1)))
            y_perm = np.asarray(model.predict(Xp))
            for t in range(nt):
                r2p = _r2_1d(y[:, t], y_perm[:, t])
                r2_vals_t[t].append(r2p)
        for t in range(nt):
            r2a = np.array(r2_vals_t[t])
            dr2 = r2_base_t[t] - r2a
            rows.append(
                dict(
                    w_start=int(s),
                    w_end=int(e),
                    center_idx=int(c),
                    f_center_GHz=float(faxis[c]),
                    target_idx=int(t),
                    target_label=str(depth_labels[t]),
                    r2_mean=float(r2a.mean()),
                    r2_std=float(r2a.std(ddof=1) if r2a.size > 1 else 0.0),
                    delta_r2_mean=float(dr2.mean()),
                    delta_r2_std=float(dr2.std(ddof=1) if dr2.size > 1 else 0.0),
                )
            )

    df = pd.DataFrame(rows).sort_values(["target_idx", "f_center_GHz"]).reset_index(drop=True)
    df.attrs["r2_baseline_per_target"] = {depth_labels[t]: float(r2_base_t[t]) for t in range(nt)}
    return df


# ----------------------------- Plot (Save) -----------------------------

# CHANGED: 4 lines (one per depth), no markers, fonts: title/axes=18, ticks=14, y-axis ≥ 0
def plot_multi_delta_r2_save(
    df_long: pd.DataFrame,
    file_path: str,
    *,
    title: str = "WSN+RF SWC ΔR² Over permuted windows feature importance",
    y_label: str = "ΔR² (baseline − permuted)",
    x_label: str = "Frequency (GHz)",
    dpi: int = 200,
) -> str:
    """Plot ΔR² lines for each target (semi-transparent, thin lines, no markers) and save.

    Args:
        df_long: Long-format DataFrame from delta_r2_by_window_permute_multi.
        file_path: Output image path.
        title: Figure title.
        y_label: Y-axis label.
        x_label: X-axis label.
        dpi: Figure DPI.

    Returns:
        Saved image path.
    """
    out_dir = os.path.dirname(file_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    # Plot with transparency and thinner lines
    for lbl, g in df_long.groupby("target_label", sort=False):
        ax.plot(
            g["f_center_GHz"].to_numpy(),
            g["delta_r2_mean"].to_numpy(),
            linewidth=1,
            alpha=0.5,
            label=lbl,
        )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, which="both", axis="both")
    xticks = np.linspace(0.4, 2.0, num=9)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks], fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(df_long["target_label"].unique().tolist(), fontsize=12, title="Depth", title_fontsize=12)
    fig.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_path



# ----------------------------- EDA JSON -----------------------------

# NEW: per-depth EDA with top windows for each depth
def summarize_multi_delta_r2_eda(
    df_long: pd.DataFrame,
    *,
    who: str = "Vincent Filardi",
    what: str = "ΔR² spectral permutation per depth with sliding windows",
    where: str = "Frequency-domain model evaluation",
    why: str = "Localize frequency bands that drive SWC prediction at each depth",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Return JSON summary with per-depth baselines and top-impact windows.

    Args:
        df_long: Long-format DataFrame from delta_r2_by_window_permute_multi.
        who: Analyst identifier.
        what: Analysis description.
        where: Context of analysis.
        why: Purpose of analysis.
        top_k: Top windows per depth by ΔR².

    Returns:
        JSON-like dict with meta, baselines, per-depth stats, and interpretation.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    base = df_long.attrs.get("r2_baseline_per_target", {})
    per_depth = {}
    for lbl, g in df_long.groupby("target_label", sort=False):
        top = g.nlargest(top_k, "delta_r2_mean")[["f_center_GHz", "delta_r2_mean", "delta_r2_std", "w_start", "w_end"]]
        per_depth[lbl] = {
            "mean_delta_r2": float(g["delta_r2_mean"].mean()),
            "median_delta_r2": float(g["delta_r2_mean"].median()),
            "top_windows": top.to_dict(orient="records"),
            "peak_window": g.loc[g["delta_r2_mean"].idxmax()][
                ["f_center_GHz", "delta_r2_mean", "delta_r2_std", "w_start", "w_end"]
            ].to_dict(),
        }

    return {
        "who": who,
        "what": what,
        "when": ts,
        "where": where,
        "why": why,
        "r2_baseline_per_depth": base,
        "summary_per_depth": per_depth,
        "interpretation": (
            "Higher ΔR² indicates that scrambling the window disrupts depth-specific associations. "
            "Peaks mark bands where the model relies most on spectral structure for that depth."
        ),
    }


# ----------------------------- I/O Helpers -----------------------------

# NEW: save DataFrame + JSON
def save_multi_outputs(
    df_long: pd.DataFrame,
    eda: Dict[str, Any],
    out_dir: str,
    prefix: str = "delta_r2_perm_multi",
) -> Dict[str, str]:
    """Save CSV and JSON EDA artifacts."""
    _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    json_path = os.path.join(out_dir, f"{prefix}.json")
    df_long.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)
    return {"csv": csv_path, "json": json_path}


# ----------------------------- Wrapper -----------------------------

# NEW: one-call pipeline for full EDA with 4-line plot
def run_delta_r2_permutation_multi(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    window_size: int = 128,
    stride: int = 32,
    f_min_ghz: float = 0.4,
    f_max_ghz: float = 2.0,
    n_permutations: int = 8,
    random_state: int = 42,
    out_dir: str = "./delta_r2_multi_artifacts",
    prefix: str = "delta_r2_perm_multi",
    depth_labels: Optional[List[str]] = ("10 cm", "20 cm", "30 cm", "40 cm"),
    dpi: int = 200,
) -> Dict[str, Any]:
    """Compute ΔR² per window×depth, save CSV/JSON/PNG, and return artifacts."""
    df_long = delta_r2_by_window_permute_multi(
        model,
        X_test,
        y_test,
        window_size=window_size,
        stride=stride,
        f_min_ghz=f_min_ghz,
        f_max_ghz=f_max_ghz,
        n_permutations=n_permutations,
        random_state=random_state,
        depth_labels=list(depth_labels) if depth_labels else None,
    )
    eda = summarize_multi_delta_r2_eda(df_long)
    paths = save_multi_outputs(df_long, eda, out_dir=out_dir, prefix=prefix)
    png_path = os.path.join(out_dir, f"{prefix}.png")
    plot_multi_delta_r2_save(df_long, png_path, title="WSN+RF SWC ΔR² over Permuted Windows by Depth", dpi=dpi)
    return {"df": df_long, "eda": eda, "paths": {**paths, "png": png_path}}

def rf_feature_importance_df(
    model,
    *,
    f_min_ghz: float = 0.4,
    f_max_ghz: float = 2.0,
) -> pd.DataFrame:
    """Return per-feature RF importances with frequency axis.

    Args:
        model: Fitted sklearn RandomForestRegressor exposing .feature_importances_.
        f_min_ghz: Minimum frequency label for feature 0.
        f_max_ghz: Maximum frequency label for last feature.

    Returns:
        DataFrame with ['feature_idx','f_GHz','rf_importance','rf_importance_norm'].
    """
    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        raise AttributeError("Model lacks 'feature_importances_'. Ensure it's a fitted RandomForestRegressor.")
    fi = np.asarray(fi, dtype=float).ravel()
    n = fi.shape[0]
    faxis = _freq_axis(n, f_min_ghz, f_max_ghz)
    df = pd.DataFrame(
        {
            "feature_idx": np.arange(n, dtype=int),
            "f_GHz": faxis.astype(float),
            "rf_importance": fi.astype(float),
        }
    ).sort_values("f_GHz", ascending=True, ignore_index=True)
    denom = max(1e-12, (df["rf_importance"].max() - df["rf_importance"].min()))
    df["rf_importance_norm"] = (df["rf_importance"] - df["rf_importance"].min()) / denom
    return df
def plot_rf_importance_save(
    df_imp: pd.DataFrame,
    file_path: str,
    *,
    title: str = "WSN+RF SWC RF Feature Importance",
    y_label: str = "RF feature importance",
    x_label: str = "Frequency (GHz)",
    use_normalized: bool = False,
    dpi: int = 200,
) -> str:
    """Plot per-feature RF importances aligned to 0.4–2.0 GHz and save.

    Args:
        df_imp: DataFrame from rf_feature_importance_df.
        file_path: Output image path.
        title: Plot title.
        y_label: Y-axis label.
        x_label: X-axis label.
        use_normalized: If True, plot normalized importance (0..1).
        dpi: Figure DPI.

    Returns:
        Saved image path.
    """
    _ensure_dir(os.path.dirname(file_path) or ".")
    x = df_imp["f_GHz"].to_numpy()
    y = (df_imp["rf_importance_norm"] if use_normalized else df_imp["rf_importance"]).to_numpy()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=1.5, alpha=0.7)  # no markers

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, which="both", axis="both")
    xticks = np.linspace(0.4, 2.0, num=9)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks], fontsize=14)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_path


# ----------------------------- EDA JSON -----------------------------

# NEW: concise EDA summary of RF per-feature importances
def summarize_rf_importance_eda(
    df_imp: pd.DataFrame,
    *,
    top_k: int = 10,
    who: str = "Vincent Filardi",
    what: str = "Per-feature RF importance over frequency",
    where: str = "Frequency-domain model evaluation",
    why: str = "Assess intrinsic model importance vs. ΔR² sensitivity",
) -> Dict[str, Any]:
    """Produce JSON with global stats and top-K frequency peaks.

    Args:
        df_imp: DataFrame from rf_feature_importance_df.
        top_k: Number of top features to report.
        who: Analyst identifier.
        what: Analysis descriptor.
        where: Context.
        why: Purpose.

    Returns:
        JSON-like dict with stats and top features.
    """
    top = df_imp.nlargest(top_k, "rf_importance")[["feature_idx", "f_GHz", "rf_importance", "rf_importance_norm"]]
    return {
        "who": who,
        "what": what,
        "where": where,
        "why": why,
        "stats": {
            "min": float(df_imp["rf_importance"].min()),
            "max": float(df_imp["rf_importance"].max()),
            "mean": float(df_imp["rf_importance"].mean()),
            "median": float(df_imp["rf_importance"].median()),
        },
        "top_features": top.to_dict(orient="records"),
    }


# ----------------------------- Wrapper -----------------------------

# NEW: one-call pipeline for RF importance on best model
def run_rf_importance(
    model,
    *,
    f_min_ghz: float = 0.4,
    f_max_ghz: float = 2.0,
    out_dir: str = "./rf_importance_best",
    prefix: str = "rf_importance",
    dpi: int = 200,
    use_normalized: bool = False,
) -> Dict[str, Any]:
    """Compute RF importances, save CSV/JSON/PNG, and return artifacts.

    Args:
        model: Fitted RandomForestRegressor.
        f_min_ghz: Minimum frequency label.
        f_max_ghz: Maximum frequency label.
        out_dir: Output directory.
        prefix: Filename prefix.
        dpi: Plot DPI.
        use_normalized: Plot normalized importances.

    Returns:
        Dict with DataFrame, EDA JSON, and paths to saved artifacts.
    """
    df_imp = rf_feature_importance_df(model, f_min_ghz=f_min_ghz, f_max_ghz=f_max_ghz)
    eda = summarize_rf_importance_eda(df_imp)
    _ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    json_path = os.path.join(out_dir, f"{prefix}.json")
    png_path = os.path.join(out_dir, f"{prefix}.png")

    df_imp.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)

    plot_rf_importance_save(
        df_imp,
        file_path=png_path,
        title="WSN+RF SWC RF Feature Importance",
        y_label="RF feature importance" if not use_normalized else "RF feature importance (normalized)",
        x_label="Frequency (GHz)",
        use_normalized=use_normalized,
        dpi=dpi,
    )
    return {"df": df_imp, "eda": eda, "paths": {"csv": csv_path, "json": json_path, "png": png_path}}
