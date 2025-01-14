import torch
import torch.nn.functional as F
from datetime import datetime


import utils

from .reconstruction_loss import ReconstructionLoss
import wandb
logger = utils.log.getLogger(__name__)

def run_svdvals(X, double=False):
    global failed_svd_counter
        
    if double:
        X = X.double()
    
    try:
        sigmas = torch.linalg.svdvals(X, driver='gesvd')
        failed_svd_counter = 0
    except torch._C._LinAlgError as e:
        logger.critical(f'SVD failed: {e}')
        failed_svd_counter += 1
        
        if torch.isnan(X).any():
            logger.warning(f'NaN in X')
        if torch.isinf(X).any():
            logger.warning(f'Inf in X')
        try:
            cond = torch.linalg.cond(X)
            logger.warning(f'Cond {cond}')
        except RuntimeError:
            logger.warning(f'Could not precompute cond number')
        if failed_svd_counter > 10:
            raise
        batch_shape = X.shape[:-2]
        sigma_shape = min(*X.shape[-2:])
        sigmas = torch.zeros(batch_shape + (sigma_shape,), device=X.device, dtype=X.dtype) * X.sum()  # dummy; multiply by X.sum() to make sure grads are attached
    return sigmas   
        
class TrackSVDValsSampled(ReconstructionLoss):
    def __init__(self, cfg, model, double=False, norm=None):
        super().__init__(cfg, model)
        self.dof = self.cfg.GWM.DOF
        self.alpha = 0.
        self.double = double
        self.norm = norm

        self.total_time = 0
        self.n = 0

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        self.training = train
        flow = self.process_flow(sample, flow)
        self.it = it
        self._extra_losses = []

        # [b, f, N, 2]
        tracks = sample['track'].to(mask_softmaxed.device)  # Should be noop

        # [b, f, N, 1]
        vis = sample['vis'].to(mask_softmaxed.device) # Should be noop

        # [b, 1, N, 2]
        coord = sample['loc'].to(mask_softmaxed.device) # Should be noop
        
        
        # print(coord.shape, vis.shape, tracks.shape)
        b, f, N, d = tracks.shape
        K = mask_softmaxed.shape[1]
        # f = vis.shape[1]

        start = datetime.now()
        masks = F.grid_sample(mask_softmaxed, coord, align_corners=False)  # b, k, 1, N
        masks = masks.clamp(1e-6, 1-1e-6)


        D = f * d
        p = tracks.permute(0, 3, 1, 2).reshape(b, 1, D, N)  # [b, f, N, 2] -> [b, 2, f, N] -> [b, 1, 2f, N]
        P = p * masks  # b k D N

        logger.info_once(f'P {P.shape}')
        sigmas = run_svdvals(P, double=self.double)  # b, K, D
        logger.info_once(f'sigmas {sigmas.shape}')


        loss = sigmas[..., self.dof:].sum(-1).mean(-1)
        loss = loss.mean().to(mask_softmaxed.dtype)

        loss = loss.mean()
        self.total_time += (datetime.now() - start).total_seconds()
        self.n += 1
        if self.n % 100 == 0:
            logger.info(f'Average loss time {self.total_time / self.n}')
        return loss
