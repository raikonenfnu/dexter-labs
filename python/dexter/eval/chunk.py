"""Score predicted action chunks against demonstrated ones.

Two reporting rules, both learned the hard way:

* **Always show a trivial baseline.** For a velocity action space, commanding
  zero scores well simply because teleoperation is often near-stationary. An
  error figure without that number beside it cannot be interpreted.
* **Report correlation as a distribution.** Flow-matching policies sample one
  plausible chunk, which need not be the one this operator chose. A pooled
  scalar cannot distinguish "usually tracks, occasionally picks another valid
  strategy" from "never tracks".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChunkScore:
    """Predicted vs demonstrated chunks, ``[N, horizon, dim]``."""

    predicted: np.ndarray
    truth: np.ndarray
    motion_dims: slice = slice(0, 7)   # joints; the last dim is the gripper

    @property
    def mae(self) -> float:
        return float(np.abs(self.predicted - self.truth)[..., self.motion_dims].mean())

    @property
    def baseline_mae(self) -> float:
        """Error of commanding zero motion."""
        return float(np.abs(self.truth[..., self.motion_dims]).mean())

    @property
    def correlation(self) -> float:
        a = self.predicted[..., self.motion_dims].ravel()
        b = self.truth[..., self.motion_dims].ravel()
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    @property
    def per_sample_correlation(self) -> np.ndarray:
        out = []
        for pred, true in zip(self.predicted, self.truth):
            a, b = pred[..., self.motion_dims].ravel(), true[..., self.motion_dims].ravel()
            out.append(np.nan if a.std() < 1e-9 or b.std() < 1e-9
                       else np.corrcoef(a, b)[0, 1])
        return np.asarray(out, dtype=float)

    def report(self) -> str:
        per = self.per_sample_correlation
        per = per[~np.isnan(per)]
        improvement = self.baseline_mae / max(self.mae, 1e-9)
        return "\n".join([
            f"samples {len(self.truth)}  horizon {self.truth.shape[1]}",
            f"  MAE            {self.mae:.4f}   do-nothing {self.baseline_mae:.4f}"
            f"   ({improvement:.2f}x {'better' if improvement > 1 else 'worse'})",
            f"  correlation    {self.correlation:+.3f} pooled,"
            f" {np.median(per):+.3f} median per sample",
            f"  tracking       {(per > 0.5).mean() * 100:.0f}% of samples at r>0.5,"
            f" {(per < 0).mean() * 100:.0f}% uncorrelated",
        ])


def score_chunks(predicted: list[np.ndarray], truth: list[np.ndarray],
                 motion_dims: slice = slice(0, 7)) -> ChunkScore:
    return ChunkScore(np.stack(predicted), np.stack(truth), motion_dims)
