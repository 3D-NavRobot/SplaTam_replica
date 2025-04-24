# utils/adaptive_prune.py
import torch
from torch.optim import Adam

class AdaptivePruner:
    """
    Adaptive Gaussian pruning that trades quality for speed on-the-fly.
    - Only prunes tensors whose first dimension == #Gaussians
    - Safely rebuilds Adam to match the new params
    """
    def __init__(
        self,
        *,
        keep_ratio_start: float = 0.6,
        keep_ratio_min:   float = 0.15,
        decay_iters:      int   = 8000,
        alpha: float = 1.0,      # weight for screen radius
        beta:  float = 0.5,      # weight for colour residual
        gamma: float = 0.2,      # weight for lifetime
        ema_decay: float = 0.9,
        lr_dict: dict,           # mapping param-name → lr for Adam
        device="cuda",
    ):
        self.k0, self.kmin, self.T = keep_ratio_start, keep_ratio_min, decay_iters
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ema_decay = ema_decay
        self.device = device
        self.lr_dict = lr_dict

        self.iter = 0
        self.ema_r = torch.tensor(1.0, device=device)
        self.ema_c = torch.tensor(0.1, device=device)
        self.birth_iter = None    # to be init on first call

    def __call__(self, params, variables, loss_rgb, seen_mask, optimizer):
        """
        Args:
            params   : dict of torch.nn.Parameter (each [N, …] or [1, …])
            variables: dict of torch.Tensor helper buffers
            loss_rgb : (N,) L1 error per Gaussian
            seen_mask: (N,) bool mask of visible Gaussians
            optimizer: existing Adam; will be replaced if pruning happens
        Returns:
            params, variables, optimizer
        """
        # --- init birth iters ---
        N = params["means3D"].shape[0]
        if self.birth_iter is None:
            self.birth_iter = torch.zeros(N, dtype=torch.int32, device=self.device)

        # --- score components ---
        # 1) screen‐radius
        r2D = variables["max_2D_radius"].clone()
        r2D = r2D / r2D.max().clamp_min(1.0)

        # 2) colour residual
        if loss_rgb.numel() != N:
            loss_rgb = torch.zeros(N, device=self.device)
        c = loss_rgb.detach()
        self.ema_c = self.ema_decay * self.ema_c + (1 - self.ema_decay) * (c.mean() if c.numel()>0 else self.ema_c)
        c_res = c / (self.ema_c + 1e-6)

        # 3) age
        age = (self.iter - self.birth_iter).float()
        age = age / age.max().clamp_min(1.0)

        # composite
        score = self.alpha*r2D + self.beta*c_res + self.gamma*age
        score[~seen_mask] += 0.25

        # --- decide keep count k ---
        it = torch.tensor(float(self.iter), device=self.device)
        kr = torch.maximum(torch.tensor(self.kmin, device=self.device),
                           self.k0 * torch.exp(-it/self.T)).item()
        k = max(min(int(kr * N), N), 1)

        keep_idx = torch.topk(-score, k).indices

        # --- prune only per-Gaussian tensors in params & variables ---
        # params: any Tensor whose first dim == N
        for name, p in list(params.items()):
            if isinstance(p, torch.nn.Parameter) and p.shape[0] == N:
                params[name] = torch.nn.Parameter(p[keep_idx].detach().clone().requires_grad_(True))

        # variables: any 1-D Tensor length N
        for name, v in list(variables.items()):
            if isinstance(v, torch.Tensor) and v.ndim==1 and v.shape[0]==N:
                variables[name] = v[keep_idx].clone()

        # prune birth record
        self.birth_iter = self.birth_iter[keep_idx]

        # --- rebuild Adam with correct param-groups ---
        new_groups = []
        for name, p in params.items():
            lr = self.lr_dict.get(name, 0.0)
            new_groups.append({'params':[p], 'lr': lr, 'name': name})
        optimizer = Adam(new_groups)

        # --- bookkeeping ---
        self.iter += 1
        self.ema_r = self.ema_decay * self.ema_r + (1 - self.ema_decay) * r2D.mean()

        return params, variables, optimizer
