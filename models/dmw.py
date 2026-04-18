import torch
import torch.nn as nn
from mmengine.registry import MODELS


@MODELS.register_module()
class DMWModule(nn.Module):
    """
    Dynamic Motion Weighting (DMW)
    Inputs:
      - f_prev: [B, C]
      - f_this: [B, C]
      - delta_base: [B, 4]   (dx, dy, dz, dyaw(deg))
    Outputs:
      - delta_dmw: [B, 4]
      - f_enhanced: [B, C]
      - w: [B, 4]
    """

    def __init__(self,
                 feat_dim: int = 256,
                 hidden: int = 256,
                 lambda_motion: float = 1.0,
                 lambda_feat: float = 0.5,
                 per_dof_weight: bool = True,
                 init_w_bias: float = -2.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.lambda_motion = float(lambda_motion)
        self.lambda_feat = float(lambda_feat)
        self.per_dof_weight = bool(per_dof_weight)

        w_out = 4 if self.per_dof_weight else 1

        self.mlp_res = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 4),
        )
        self.mlp_w = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, w_out),
        )
        self.mlp_feat = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, self.feat_dim),
        )

        # safe init: start as identity
        nn.init.zeros_(self.mlp_res[-1].weight)
        nn.init.zeros_(self.mlp_res[-1].bias)

        nn.init.zeros_(self.mlp_w[-1].weight)
        nn.init.constant_(self.mlp_w[-1].bias, init_w_bias)

        nn.init.zeros_(self.mlp_feat[-1].weight)
        nn.init.zeros_(self.mlp_feat[-1].bias)

    def forward(self, f_prev: torch.Tensor, f_this: torch.Tensor, delta_base: torch.Tensor):
        dF = torch.abs(f_this - f_prev)   # [B,C]
        delta_res = self.mlp_res(dF)      # [B,4]
        w = torch.sigmoid(self.mlp_w(dF)) # [B,4] or [B,1]
        if w.shape[-1] == 1:
            w = w.expand(-1, 4)

        delta_dmw = delta_base + self.lambda_motion * w * delta_res
        f_enhanced = f_this + self.lambda_feat * self.mlp_feat(dF)
        return delta_dmw, f_enhanced, w
