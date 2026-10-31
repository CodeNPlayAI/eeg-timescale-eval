# src/common/granger.py
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def granger_var_aic(x: np.ndarray, y: np.ndarray, maxlags: int = 20) -> dict:
    """
    Fit VAR([x,y]) with lag selected by AIC. Return predictive-direction metrics.

    Returns:
      {
        "lag": int,
        "x->y_p": float,
        "y->x_p": float
      }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    df = pd.DataFrame({"x": x, "y": y}).dropna()
    maxlags = min(maxlags, len(df) // 5)
    if maxlags < 1:
        return {"lag": None, "x->y_p": float("nan"), "y->x_p": float("nan")}

    model = VAR(df)
    res = model.fit(maxlags=maxlags, ic="aic")
    lag = int(res.k_ar)

    # statsmodels: test_causality(causing, caused)
    p_x_to_y = float(res.test_causality(caused="y", causing=["x"], kind="f").pvalue)
    p_y_to_x = float(res.test_causality(caused="x", causing=["y"], kind="f").pvalue)
    return {"lag": lag, "x->y_p": p_x_to_y, "y->x_p": p_y_to_x}
