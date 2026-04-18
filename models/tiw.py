import torch
import torch.nn as nn
import math
from mmengine.registry import MODELS


@MODELS.register_module()
class TIWModule(nn.Module):
    """
    Temporal Importance Weighting (TIW) for inference smoothing with gating.
    """

    def __init__(self,
                 feat_dim: int = 256,
                 L: int = 5,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 sigma: float = 1.0,
                 use_delta_norm: bool = True):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.L = int(L)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.use_delta_norm = bool(use_delta_norm)

    def _norm_delta(self, d: torch.Tensor):
        if not self.use_delta_norm:
            return d
        scale = d.new_tensor([0.5, 0.5, 0.2, 10.0])  # dx dy dz dyaw(deg)
        return d / scale

    def forward(self, f_cur, delta_cur, hist_feats, hist_deltas):
        # f_cur: [B,C], delta_cur: [B,4], hist_feats: [B,T,C], hist_deltas: [B,T,4]
        B, T, C = hist_feats.shape
        assert C == self.feat_dim, f"feat_dim mismatch: {C} vs {self.feat_dim}"

        sim = torch.einsum('bc,btc->bt', f_cur, hist_feats) / math.sqrt(C)
        alpha_w = torch.softmax(sim, dim=-1)

        d0 = self._norm_delta(delta_cur)
        dH = self._norm_delta(hist_deltas)
        diff = dH - d0[:, None, :]
        dist2 = (diff * diff).sum(dim=-1)
        beta_w = torch.exp(-dist2 / (self.sigma * self.sigma + 1e-6))

        w = self.alpha * alpha_w + self.beta * beta_w
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        delta_tiw = torch.einsum('bt,btj->bj', w, hist_deltas)
        return delta_tiw, w
