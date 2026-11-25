"""Ordered config 41 features from prepared frequency mean sweep scans."""

import numpy as np
import pandas as pd

from Data_process import height_wavelet_transform
from wavelet_rf_split import HEIGHT


SCHEMA_VERSION = 1
WAVELET_SETTINGS = {"J": 6, "Q": 32, "max_order": 2}
SCAN_COLUMNS = tuple((HEIGHT, f"freq_mean_sweep_{i}.1") for i in range(4096))
DEPTHS_CM = (10, 20, 30, 40)
TARGET_COLUMNS = tuple(("y", f"avgdepth{i}") for i in range(1, 5))


def extract_features(scans):
    """Return the research fitter's ordered feature frame for unlabeled or labeled scans."""
    if not isinstance(scans.columns, pd.MultiIndex) or scans.columns.nlevels != 2:
        raise ValueError("Expected two header levels for config 41 scans.")
    if scans.columns.has_duplicates or scans.empty:
        raise ValueError("Expected unique scan columns and at least one sample.")
    height_columns = tuple(col for col in scans.columns if col[0].startswith("height"))
    if height_columns != SCAN_COLUMNS:
        raise ValueError(
            "Expected only height_2 with all 4096 freq_mean_sweep samples "
            "in original frequency order (freq_mean_sweep_0.1 through "
            "freq_mean_sweep_4095.1)."
        )
    try:
        signals = scans.loc[:, list(SCAN_COLUMNS)].to_numpy(dtype=np.complex128)
    except (ValueError, TypeError) as exc:
        raise ValueError("Height_2 samples must be numeric complex values.") from exc
    if not np.isfinite(signals).all():
        raise ValueError("Height_2 contains missing or nonfinite samples.")
    constant = (np.ptp(signals.real, axis=1) == 0) | (np.ptp(signals.imag, axis=1) == 0)
    if constant.any():
        positions = np.flatnonzero(constant)[:5].tolist()
        raise ValueError(
            f"Height_2 has a zero-range real or imaginary component at row positions {positions}. "
            "Supply scans with variation in both components for row-wise min-max normalization."
        )

    prepared = pd.DataFrame(signals, columns=pd.MultiIndex.from_tuples(SCAN_COLUMNS))
    scattered = height_wavelet_transform(prepared, [HEIGHT], **WAVELET_SETTINGS)
    coefficients = scattered.to_numpy()
    if not np.isfinite(coefficients).all():
        raise ValueError("Height_2 wavelet scattering produced nonfinite features.")

    # The research loader parses each coefficient as complex and interleaves
    # its real and imaginary parts, including the zero imaginary part.
    features = np.zeros((len(scans), 2 * coefficients.shape[1]), dtype=np.float64)
    features[:, ::2] = coefficients
    columns = pd.MultiIndex.from_tuples([
        (group, f"{component}_{name}")
        for group, name in scattered.columns
        for component in ("re", "im")
    ])
    return pd.DataFrame(features, index=scans.index, columns=columns)
