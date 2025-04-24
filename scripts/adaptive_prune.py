# utils/adaptive_prune.py
import torch
from torch.optim import Adam

class AdaptivePruner:
    """
    CPU-backed, safe on-the-fly Gaussian pruning.
    Moves only the per-Gaussian buffers to CPU for indexing,
    then brings them back to GPU, and rebuilds Adam.
    """
    def __init__(
        self,
        *,
        keep_ratio_start: float,
        keep_ratio_min:   float,
        decay_iters:      int,
        alpha: float,
        beta:  float,
        gamma: float,
        ema_decay: float,
        lrs: dict,           # mapping from param-name to lr
        device="cuda",
    ):
        self.k0, self.kmin, self.T = keep_ratio_start, keep_ratio_min, decay_iters
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ema_decay = ema_decay
        self.lrs = lrs
        self.device = device

        self.iter = 0
        self.ema_r = torch.tensor(1.0, device=device)
        self.ema_c = torch.tensor(0.1, device=device)
        self.birth_iter = None

    def __call__(self, params, variables, loss_rgb, seen_mask, optimizer):
        N = params["means3D"].shape[0]
        if self.birth_iter is None:
            self.birth_iter = torch.zeros(N, dtype=torch.int32, device=self.device)

        # 1) screen‐space radius
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

        # composite score
        score = self.alpha*r2D + self.beta*c_res + self.gamma*age
        score[~seen_mask] += 0.25

        # decide how many to keep
        it = torch.tensor(float(self.iter), device=self.device)
        kr = torch.maximum(
            torch.tensor(self.kmin, device=self.device),
            self.k0 * torch.exp(-it/self.T)
        ).item()
        k = max(min(int(kr * N), N), 1)
        keep_idx = torch.topk(-score, k).indices

        # --- do all indexing on CPU to catch bad indices early ---
        keep_cpu = keep_idx.cpu()

        # prune params
        new_params = {}
        for name, p in params.items():
            if isinstance(p, torch.nn.Parameter) and p.shape[0] == N:
                cpu_tensor = p.detach().cpu()
                sliced = cpu_tensor[keep_cpu]
                new_params[name] = torch.nn.Parameter(sliced.to(self.device).clone().requires_grad_(True))
            else:
                new_params[name] = p
        params.clear(); params.update(new_params)

        # prune variables
        for name, v in list(variables.items()):
            if isinstance(v, torch.Tensor) and v.ndim==1 and v.shape[0]==N:
                cpu_v = v.detach().cpu()
                variables[name] = cpu_v[keep_cpu].to(self.device)
            # else leave it alone

        # prune birth records
        self.birth_iter = self.birth_iter[keep_idx]

        # rebuild optimizer
        groups = []
        for name, p in params.items():
            lr = self.lrs.get(name, 0.0)
            groups.append({'params':[p], 'lr':lr, 'name':name})
        optimizer = Adam(groups)

        # bookkeeping
        self.iter += 1
        self.ema_r = self.ema_decay*self.ema_r + (1-self.ema_decay)*r2D.mean()

        return params, variables, optimizer
