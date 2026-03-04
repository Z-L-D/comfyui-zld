"""
Spectrum: Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration
Core algorithm implementation.

Based on: Han et al., "Adaptive Spectral Feature Forecasting for Diffusion Sampling
Acceleration" (CVPR 2026)  —  https://arxiv.org/abs/2603.01623

Core math:
  1. Map diffusion timesteps to [-1, 1] (Chebyshev domain)
  2. Evaluate Chebyshev polynomial bases T_0..T_{m-1}
  3. Fit coefficients via ridge regression on cached observations
  4. Forecast features at future timesteps using the fitted polynomial
  5. Optionally mix with linear extrapolation for robustness (w < 1)
"""

import torch
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Chebyshev basis evaluation
# ---------------------------------------------------------------------------

def chebyshev_basis(t: torch.Tensor, m: int) -> torch.Tensor:
    """
    T_0=1, T_1=t, T_j = 2*t*T_{j-1} - T_{j-2}

    Args:
        t: shape (n,), values in [-1, 1]
        m: number of bases
    Returns:
        B: shape (n, m), B[i,j] = T_j(t[i])
    """
    n = t.shape[0]
    B = torch.zeros(n, m, device=t.device, dtype=t.dtype)
    if m >= 1:
        B[:, 0] = 1.0
    if m >= 2:
        B[:, 1] = t
    for j in range(2, m):
        B[:, j] = 2.0 * t * B[:, j - 1] - B[:, j - 2]
    return B


# ---------------------------------------------------------------------------
# Ridge regression solver
# ---------------------------------------------------------------------------

def ridge_solve(B: torch.Tensor, F: torch.Tensor, lam: float) -> torch.Tensor:
    """C = (B^T B + λI)^{-1} B^T F"""
    m = B.shape[1]
    BtB = B.T @ B
    BtB.diagonal().add_(lam)
    return torch.linalg.solve(BtB, B.T @ F)


# ---------------------------------------------------------------------------
# Step scheduler
# ---------------------------------------------------------------------------

class SpectrumScheduler:
    """
    Decides which sampling steps run the denoiser (full-compute)
    vs. use forecasted output (skip).

    window_size  N : initial gap between full-compute steps
    flex_window  α : controls window growth (<1 → windows grow over time,
                     meaning later steps are skipped more aggressively)
    first_enhance  : forced full-compute steps at the start to bootstrap cache
    """

    def __init__(self, num_steps: int, window_size: int = 2,
                 flex_window: float = 0.75, first_enhance: int = 4):
        self.num_steps = num_steps
        self.window_size = window_size
        self.flex_window = flex_window
        self.first_enhance = first_enhance
        self._full = self._build()

    def _build(self) -> set:
        full = set()
        # Bootstrap phase
        for i in range(min(self.first_enhance, self.num_steps)):
            full.add(i)
        # Windowed phase
        pos = max(self.first_enhance, 0)
        full.add(min(pos, self.num_steps - 1))
        win = float(self.window_size)
        while pos < self.num_steps:
            gap = max(1, int(round(win)))
            pos += gap
            if pos < self.num_steps:
                full.add(pos)
            if self.flex_window > 0:
                win = win / self.flex_window
            else:
                break
        full.add(self.num_steps - 1)
        return full

    def is_full(self, step: int) -> bool:
        return step in self._full

    @property
    def compression(self) -> float:
        return self.num_steps / max(len(self._full), 1)


# ---------------------------------------------------------------------------
# Per-slot Chebyshev forecaster
# ---------------------------------------------------------------------------

class SpectrumForecaster:
    """
    Caches (t, feature) observations from full-compute steps and
    predicts features on skip steps via Chebyshev ridge regression.

    m   : number of Chebyshev bases (default 4)
    lam : ridge regularisation      (default 0.1)
    w   : Chebyshev/linear-extrap mix weight; 1.0 = pure Chebyshev,
          recommended 0.5–1.0
    """

    def __init__(self, m: int = 4, lam: float = 0.1, w: float = 0.5):
        self.m = m
        self.lam = lam
        self.w = w
        self.ts: List[float] = []
        self.feats: List[torch.Tensor] = []
        self._coeff: Optional[torch.Tensor] = None
        self._nbases = 0
        self._shape = None
        self._dirty = True

    def reset(self):
        self.ts.clear()
        self.feats.clear()
        self._coeff = None
        self._dirty = True

    def push(self, t: float, feat: torch.Tensor):
        self.ts.append(t)
        self.feats.append(feat.detach())
        self._shape = feat.shape
        self._dirty = True

    @property
    def n(self) -> int:
        return len(self.ts)

    def _fit(self):
        if self.n < 2:
            return
        dev = self.feats[0].device
        dt = self.feats[0].dtype
        t_vec = torch.tensor(self.ts, device=dev, dtype=dt)
        nb = min(self.m, self.n)
        B = chebyshev_basis(t_vec, nb)
        F = torch.stack([f.reshape(-1) for f in self.feats])
        self._coeff = ridge_solve(B, F, self.lam)
        self._nbases = nb
        self._dirty = False

    def forecast(self, t: float) -> Optional[torch.Tensor]:
        if self.n < 2:
            return None
        if self._dirty:
            self._fit()
        if self._coeff is None:
            return None

        dev = self._coeff.device
        dt = self._coeff.dtype
        t_vec = torch.tensor([t], device=dev, dtype=dt)
        B_t = chebyshev_basis(t_vec, self._nbases)
        cheb = (B_t @ self._coeff).reshape(self._shape)

        if self.w < 1.0:
            lin = self._linear_extrap(t)
            if lin is not None:
                return self.w * cheb + (1.0 - self.w) * lin
        return cheb

    def _linear_extrap(self, t: float) -> Optional[torch.Tensor]:
        if self.n < 2:
            return None
        t0, t1 = self.ts[-2], self.ts[-1]
        f0, f1 = self.feats[-2], self.feats[-1]
        gap = t1 - t0
        if abs(gap) < 1e-12:
            return f1.clone()
        alpha = (t - t0) / gap
        return f0 + alpha * (f1 - f0)


# ---------------------------------------------------------------------------
# Top-level state container
# ---------------------------------------------------------------------------

class SpectrumState:
    """
    Holds everything for one Spectrum-accelerated generation:
    scheduler, per-slot forecasters, and the step counter.

    In model-output mode a single forecaster ("_model") predicts the
    entire denoiser output.  For block-level mode, one forecaster per
    block is created on demand via get_forecaster(block_id).
    """

    def __init__(self, num_steps: int, m: int = 4, lam: float = 0.1,
                 w: float = 0.5, window_size: int = 2,
                 flex_window: float = 0.75, first_enhance: int = 4):

        self.num_steps = num_steps
        self.m = m
        self.lam = lam
        self.w = w
        self.sched = SpectrumScheduler(
            num_steps, window_size, flex_window, first_enhance)
        self.forecasters: Dict[str, SpectrumForecaster] = {}
        self.step: int = 0

    def reset(self):
        self.step = 0
        for fc in self.forecasters.values():
            fc.reset()

    def get_forecaster(self, key: str = "_model") -> SpectrumForecaster:
        if key not in self.forecasters:
            self.forecasters[key] = SpectrumForecaster(
                m=self.m, lam=self.lam, w=self.w)
        return self.forecasters[key]

    def t_norm(self, step: Optional[int] = None) -> float:
        s = step if step is not None else self.step
        denom = max(self.num_steps - 1, 1)
        return 2.0 * s / denom - 1.0

    @property
    def is_full(self) -> bool:
        return self.sched.is_full(self.step)

    def advance(self):
        self.step += 1
