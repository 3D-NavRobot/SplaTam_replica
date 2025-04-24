import torch
from torch.optim import Adam

class AdaptivePruner:
    """
    Safe on-the-fly Gaussian pruning:
      • CPU-index per-gaussian tensors to catch OOB early.
      • Only touches tensors whose .shape[0]==N.
      • Rebuilds Adam with fresh param groups.
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
        lrs: dict,           # map param_name→lr
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
        self.birth = None

    def __call__(self, params, variables, loss_rgb, seen, optimizer):
        N = params["means3D"].shape[0]
        # init birth times
        if self.birth is None:
            self.birth = torch.zeros(N, dtype=torch.int32, device=self.device)

        # 1) screen radius
        r2D = variables["max_2D_radius"].clone()
        r2D = r2D / r2D.max().clamp_min(1.0)

        # 2) color residual
        rgb = loss_rgb.detach() if loss_rgb.numel()==N else torch.zeros(N, device=self.device)
        self.ema_c = (self.ema_decay*self.ema_c +
                      (1-self.ema_decay)*rgb.mean().clamp_min(self.ema_c))
        c_res = rgb / (self.ema_c + 1e-6)

        # 3) age
        age = (self.iter - self.birth).float()
        age = age / age.max().clamp_min(1.0)

        # composite score
        score = self.alpha*r2D + self.beta*c_res + self.gamma*age
        score[~seen] += 0.25

        # determine keep-count
        t = torch.tensor(float(self.iter), device=self.device)
        kr = max(self.kmin, self.k0*torch.exp(-t/self.T).item())
        k = min(max(int(kr * N), 1), N)

        # get keep indices
        keep = torch.topk(-score, k).indices
        keep_cpu = keep.cpu()  # <-- CPU index to catch any OOB

        # slice params
        new_params = {}
        for name,p in params.items():
            if isinstance(p, torch.nn.Parameter) and p.shape[0]==N:
                cpu = p.detach().cpu()
                sliced = cpu[keep_cpu]
                new_params[name] = torch.nn.Parameter(
                    sliced.to(self.device).clone().requires_grad_(True)
                )
            else:
                new_params[name] = p
        params.clear(); params.update(new_params)

        # slice variables
        for name,v in list(variables.items()):
            if isinstance(v, torch.Tensor) and v.ndim==1 and v.shape[0]==N:
                cpu = v.detach().cpu()
                variables[name] = cpu[keep_cpu].to(self.device)
            # else: leave alone

        # update birth times
        self.birth = self.birth[keep]

        # rebuild optimizer
        groups = []
        for nm,p in params.items():
            lr = self.lrs.get(nm, 0.0)
            groups.append({'params':[p], 'lr':lr, 'name':nm})
        optimizer = Adam(groups)

        # bookkeeping
        self.iter += 1
        self.ema_r = self.ema_decay*self.ema_r + (1-self.ema_decay)*r2D.mean()

        return params, variables, optimizer
