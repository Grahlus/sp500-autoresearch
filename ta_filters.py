"""
ta_filters.py — Technical indicator filters for TA session.
All indicators computed from close prices only (numpy/pandas, no external TA libs).
"""
import numpy as np
import pandas as pd


def bb_position(close_slice: pd.Series, window: int = 20, n_std: float = 2.0) -> float:
    """Returns position within BB bands: 0=lower, 0.5=middle, 1=upper.
    >0.80 = near/above upper band (overextended)
    <0.20 = near/below lower band (oversold)
    """
    if len(close_slice) < window:
        return 0.5
    s = close_slice.iloc[-window:]
    sma = float(s.mean())
    std = float(s.std())
    if std == 0:
        return 0.5
    upper = sma + n_std * std
    lower = sma - n_std * std
    if upper == lower:
        return 0.5
    return float((close_slice.iloc[-1] - lower) / (upper - lower))


def _sma50(close_slice: pd.Series) -> float:
    """SMA of last 50 bars in slice."""
    if len(close_slice) < 50:
        return float(close_slice.mean())
    return float(close_slice.iloc[-50:].mean())


def sma50_slope(close_slice_60d: pd.Series) -> float:
    """Returns slope of SMA50 as fractional change per day.
    slope = (sma50[T] - sma50[T-10]) / sma50[T-10] / 10
    Positive = uptrend, negative = downtrend.
    """
    if len(close_slice_60d) < 60:
        return 0.0
    sma_now  = float(close_slice_60d.iloc[-50:].mean())
    sma_prev = float(close_slice_60d.iloc[-60:-10].mean())
    if sma_prev <= 0:
        return 0.0
    return (sma_now - sma_prev) / sma_prev / 10.0


def sma50_position(close_today: float, close_slice_60d: pd.Series) -> float:
    """Returns close[T] / sma50[T] - 1.
    Positive = above SMA50, negative = below.
    """
    sma = _sma50(close_slice_60d)
    if sma <= 0:
        return 0.0
    return close_today / sma - 1.0


def detect_double_top(close_slice: pd.Series, lookback: int = 60, tolerance: float = 0.03) -> bool:
    """Returns True if a double-top (M pattern) is present.
    - Find highest peak in close_slice
    - Find second peak within tolerance% of first peak
    - Valley between them >= 5% below both peaks
    - Second peak within last 20 bars
    """
    s = close_slice.iloc[-lookback:] if len(close_slice) >= lookback else close_slice
    if len(s) < 20:
        return False
    vals = s.values

    # Find first (highest) peak — local max in first 2/3 of window
    split = max(len(vals) - 20, 10)
    first_half = vals[:split]
    if len(first_half) < 3:
        return False
    # local maxima in first half
    first_peaks = []
    for j in range(1, len(first_half) - 1):
        if first_half[j] >= first_half[j-1] and first_half[j] >= first_half[j+1]:
            first_peaks.append((j, first_half[j]))
    if not first_peaks:
        return False
    p1_idx, p1_val = max(first_peaks, key=lambda x: x[1])

    # Find second peak in last 20 bars
    second_half = vals[split:]
    second_peaks = []
    for j in range(1, len(second_half) - 1):
        if second_half[j] >= second_half[j-1] and second_half[j] >= second_half[j+1]:
            second_peaks.append((split + j, second_half[j]))
    if not second_peaks:
        return False
    p2_idx, p2_val = max(second_peaks, key=lambda x: x[1])

    # Both peaks within tolerance of each other
    if abs(p1_val - p2_val) / max(p1_val, p2_val) > tolerance:
        return False

    # Valley between peaks must be >= 5% below both peaks
    valley = vals[p1_idx:p2_idx]
    if len(valley) == 0:
        return False
    min_between = float(np.min(valley))
    threshold = min(p1_val, p2_val) * 0.95
    if min_between > threshold:
        return False

    return True


def detect_double_bottom(close_slice: pd.Series, lookback: int = 60, tolerance: float = 0.03) -> bool:
    """Returns True if a double-bottom (W pattern) is present.
    Inverse of double_top: two lows within tolerance%, rally >= 5% between them.
    Second bottom within last 20 bars.
    """
    s = close_slice.iloc[-lookback:] if len(close_slice) >= lookback else close_slice
    if len(s) < 20:
        return False
    vals = s.values

    split = max(len(vals) - 20, 10)
    first_half = vals[:split]
    if len(first_half) < 3:
        return False

    # Local minima in first half
    first_troughs = []
    for j in range(1, len(first_half) - 1):
        if first_half[j] <= first_half[j-1] and first_half[j] <= first_half[j+1]:
            first_troughs.append((j, first_half[j]))
    if not first_troughs:
        return False
    t1_idx, t1_val = min(first_troughs, key=lambda x: x[1])

    # Local minima in last 20 bars
    second_half = vals[split:]
    second_troughs = []
    for j in range(1, len(second_half) - 1):
        if second_half[j] <= second_half[j-1] and second_half[j] <= second_half[j+1]:
            second_troughs.append((split + j, second_half[j]))
    if not second_troughs:
        return False
    t2_idx, t2_val = min(second_troughs, key=lambda x: x[1])

    # Both troughs within tolerance
    if abs(t1_val - t2_val) / max(t1_val, t2_val) > tolerance:
        return False

    # Rally between troughs >= 5% above both troughs
    between = vals[t1_idx:t2_idx]
    if len(between) == 0:
        return False
    max_between = float(np.max(between))
    threshold = max(t1_val, t2_val) * 1.05
    if max_between < threshold:
        return False

    return True


def detect_triple_top(close_slice: pd.Series, lookback: int = 80, tolerance: float = 0.03) -> bool:
    """Three peaks within tolerance% of each other.
    All within last 40 bars. Valleys between peaks >= 4% below peaks.
    """
    s = close_slice.iloc[-lookback:] if len(close_slice) >= lookback else close_slice
    window = s.iloc[-40:] if len(s) >= 40 else s
    if len(window) < 10:
        return False
    vals = window.values

    # Find all local maxima
    peaks = []
    for j in range(1, len(vals) - 1):
        if vals[j] >= vals[j-1] and vals[j] >= vals[j+1]:
            peaks.append((j, vals[j]))
    if len(peaks) < 3:
        return False

    # Try all combinations of 3 peaks in order
    for a in range(len(peaks) - 2):
        for b in range(a + 1, len(peaks) - 1):
            for c in range(b + 1, len(peaks)):
                i1, v1 = peaks[a]
                i2, v2 = peaks[b]
                i3, v3 = peaks[c]
                avg_peak = (v1 + v2 + v3) / 3.0
                if avg_peak <= 0:
                    continue
                # All three within tolerance
                if (abs(v1 - avg_peak) / avg_peak > tolerance or
                        abs(v2 - avg_peak) / avg_peak > tolerance or
                        abs(v3 - avg_peak) / avg_peak > tolerance):
                    continue
                # Valleys between must be >= 4% below min peak
                min_peak = min(v1, v2, v3)
                valley12 = vals[i1:i2]
                valley23 = vals[i2:i3]
                if len(valley12) == 0 or len(valley23) == 0:
                    continue
                if (float(np.min(valley12)) > min_peak * 0.96 or
                        float(np.min(valley23)) > min_peak * 0.96):
                    continue
                return True
    return False


def detect_triple_bottom(close_slice: pd.Series, lookback: int = 80, tolerance: float = 0.03) -> bool:
    """Inverse of triple_top: three troughs within tolerance%, rallies >= 4% between them."""
    s = close_slice.iloc[-lookback:] if len(close_slice) >= lookback else close_slice
    window = s.iloc[-40:] if len(s) >= 40 else s
    if len(window) < 10:
        return False
    vals = window.values

    troughs = []
    for j in range(1, len(vals) - 1):
        if vals[j] <= vals[j-1] and vals[j] <= vals[j+1]:
            troughs.append((j, vals[j]))
    if len(troughs) < 3:
        return False

    for a in range(len(troughs) - 2):
        for b in range(a + 1, len(troughs) - 1):
            for c in range(b + 1, len(troughs)):
                i1, v1 = troughs[a]
                i2, v2 = troughs[b]
                i3, v3 = troughs[c]
                avg_trough = (v1 + v2 + v3) / 3.0
                if avg_trough <= 0:
                    continue
                if (abs(v1 - avg_trough) / avg_trough > tolerance or
                        abs(v2 - avg_trough) / avg_trough > tolerance or
                        abs(v3 - avg_trough) / avg_trough > tolerance):
                    continue
                max_trough = max(v1, v2, v3)
                rally12 = vals[i1:i2]
                rally23 = vals[i2:i3]
                if len(rally12) == 0 or len(rally23) == 0:
                    continue
                if (float(np.max(rally12)) < max_trough * 1.04 or
                        float(np.max(rally23)) < max_trough * 1.04):
                    continue
                return True
    return False
