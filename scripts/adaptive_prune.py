# utils/adaptive_prune.py  (NEW FILE)
import torch

class AdaptivePruner:
    """
    Adaptive Gaussian pruning that trades quality for speed on-the-fly.
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
        device="cuda",
    ):
        self.k0   = keep_ratio_start
        self.kmin = keep_ratio_min
        self.T    = decay_iters
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ema_decay = ema_decay
        self.device = device

        self.iter = 0
        # running statistics
        self.ema_r = torch.tensor(1.0, device=device)
        self.ema_c = torch.tensor(0.1, device=device)

        # each gaussian's 'birthday'
        self.birth_iter = None    # initialised on first call

    # ---------------------------------------------------------------------- #
    def __call__(self, params, variables, loss_rgb, seen_mask):
        """
        Args
        ----
        params      : dict – optimisation parameters (will be shrunk in-place)
        variables   : dict – helper tensors (will be shrunk in-place)
        loss_rgb    : (N,) tensor – |render-gt| L1 per-gaussian (already
                      accumulated in get_loss via means2D_gradient_accum)
        seen_mask   : (N,) bool – gaussians visible in current frame
        """
        N = params["means3D"].shape[0]
        if self.birth_iter is None:
            self.birth_iter = torch.zeros(N, dtype=torch.int32,
                                          device=self.device)

        # --- compute screen-space radius (normalised 0–1) ------------------
        r2D = variables["max_2D_radius"].clone()
        r2D = r2D / (r2D.max().clamp_min(1.))      # scale invariant

        # --- colour residual (normalised) ----------------------------------
        c_res = loss_rgb.detach()
        if c_res.numel() == 0:                      # no vis pixels this iter
            c_res = torch.zeros_like(r2D)
        # update running mean so different scenes stay comparable
        self.ema_c = self.ema_decay * self.ema_c + (1 - self.ema_decay) * (
            c_res.mean() if (c_res > 0).any() else self.ema_c)
        c_res = c_res / (self.ema_c + 1e-6)

        # --- lifetime term -------------------------------------------------
        life = (self.iter - self.birth_iter).float()
        life = life / (life.max().clamp_min(1.))

        # --- composite score ----------------------------------------------
        score = (self.alpha * r2D +
                 self.beta  * c_res +
                 self.gamma * life)

        # unseen gaussians slowly become “old” → pruned later anyway
        score[~seen_mask] += 0.25

        # keep only k lowest score
        keep_ratio = max(self.kmin,
                         self.k0 * torch.exp(-self.iter / self.T))
        k = max(int(keep_ratio * N), int(self.kmin * N))
        keep_idx = torch.topk(-score, k).indices     # negative → lowest

        # --- prune tensors -------------------------------------------------
        for kname, tensor in params.items():
            params[kname] = tensor[keep_idx]

        for vname, tensor in variables.items():
            if tensor.ndim == 1:
                variables[vname] = tensor[keep_idx]

        # update birth dates (remove dead, add new)
        self.birth_iter = self.birth_iter[keep_idx]

        # book-keeping
        self.iter += 1
        self.ema_r = self.ema_decay * self.ema_r + (1 - self.ema_decay) * r2D.mean()

        return params, variables
