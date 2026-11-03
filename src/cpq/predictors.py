from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LassoCV

MAX_PAIRS = 15


@dataclass
class GridPredictor:
    """y_hat = delta * (k0 + ko.x + sum kp * x_i x_j) with integer coefficients."""

    delta: float
    k0: int
    ko: np.ndarray
    pairs: np.ndarray
    kp: np.ndarray
    raw: tuple[float, np.ndarray, np.ndarray]

    def _eval(self, X: np.ndarray, k0, ko, kp) -> np.ndarray:
        out = k0 + X @ ko
        for (i, j), k in zip(self.pairs, kp, strict=True):
            out = out + k * X[:, i] * X[:, j]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        return self.delta * self._eval(X, self.k0, self.ko, self.kp)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Same terms without rounding, to report the effect of the grid."""
        k0, ko, kp = self.raw
        return self.delta * self._eval(X.astype(np.float64), k0, ko, kp)

    def low(self) -> int:
        """Grid index (in units of delta) below every prediction."""
        return int(self.k0 + np.minimum(self.ko, 0).sum() + np.minimum(self.kp, 0).sum())

    def floor(self, ymin: float) -> int:
        """Grid offset c0 / delta below every prediction and every value >= ymin."""
        return int(min(self.low(), np.floor(ymin / self.delta)))

    def top_pairs(self, k: int) -> GridPredictor:
        """The same predictor keeping the k pairs of largest raw coefficient."""
        keep = np.sort(np.argsort(-np.abs(self.raw[2]), kind="stable")[:k])
        k0, ko, kp = self.raw
        return GridPredictor(self.delta, self.k0, self.ko, self.pairs[keep], self.kp[keep],
                             (k0, ko, kp[keep]))  # fmt: skip

    def weights(self, X: np.ndarray, c0: int) -> np.ndarray:
        """Non-negative integer grid index of each row, (y_hat - c0 * delta) / delta."""
        return np.rint(self.predict(X) / self.delta).astype(np.int64) - c0


def fit_grid(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GridPredictor:
    """LassoCV on options and pairs of Train, at most 15 pairs, coefficients on the grid."""
    delta = (np.percentile(y, 75) - y.min()) / 128
    X = X.astype(np.float64)
    nf = X.shape[1]
    var = np.flatnonzero(X.std(axis=0) > 0)
    pairs = np.array([(i, j) for a, i in enumerate(var) for j in var[a + 1 :]], dtype=np.int64)
    pairs = pairs.reshape(-1, 2)
    P = X[:, pairs[:, 0]] * X[:, pairs[:, 1]]
    keep = P.std(axis=0) > 0
    pairs, P = pairs[keep], P[:, keep]
    M = np.hstack([X[:, var], P])
    lasso = LassoCV(cv=min(5, len(y)), random_state=seed, max_iter=20000).fit(M, y)
    co = np.zeros(nf)
    co[var] = lasso.coef_[: len(var)]
    cp = lasso.coef_[len(var) :]
    top = np.argsort(-np.abs(cp))[:MAX_PAIRS]
    top = top[cp[top] != 0]
    pairs, cp = pairs[top], cp[top]
    k0 = float(lasso.intercept_)
    return GridPredictor(
        delta=float(delta),
        k0=int(np.rint(k0 / delta)),
        ko=np.rint(co / delta),
        pairs=pairs,
        kp=np.rint(cp / delta),
        raw=(k0 / delta, co / delta, cp / delta),
    )
