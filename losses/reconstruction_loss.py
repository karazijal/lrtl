import torch
import torch.nn.functional as F
from torch import nn

import flow_reconstruction
import utils
from utils.visualisation import flow2rgb_torch

logger = utils.log.getLogger(__name__)


class ReconstructionLoss:
    def __init__(self, cfg, model):
        self.criterion = nn.MSELoss() if cfg.GWM.CRITERION == "L2" else nn.L1Loss()
        self.l1_optimize = cfg.GWM.L1_OPTIMIZE
        self.homography = cfg.GWM.HOMOGRAPHY
        self.device = model.device
        self.cfg = cfg
        self.grid_x, self.grid_y = utils.grid.get_meshgrid(cfg.GWM.RESOLUTION, model.device)
        # self.mult_flow = cfg.GWM.USE_MULT_FLOW
        self.flow_colorspace_rec = cfg.GWM.FLOW_COLORSPACE_REC
        self.flow_u_low = cfg.GWM.FLOW_CLIP_U_LOW
        self.flow_u_high = cfg.GWM.FLOW_CLIP_U_HIGH
        self.flow_v_low = cfg.GWM.FLOW_CLIP_V_LOW
        self.flow_v_high = cfg.GWM.FLOW_CLIP_V_HIGH

        self._recon_fn = self.flow_quad
        logger.info(f"Using reconstruction method {self._recon_fn.__name__}")
        self.it = 0
        self._extra_losses = []

    def __call__(self, sample, flow, masks_softmaxed, it, train=True):
        return self.loss(sample, flow, masks_softmaxed, it, train=train)

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        self.training = train
        flow = self.process_flow(sample, flow).to(mask_softmaxed.device)
        self.it = it
        self._extra_losses = []

        if self.cfg.GWM.FLOW_RES is not None:
            if flow.shape[-2:] != mask_softmaxed.shape[-2:]:
                logger.debug_once(f"Resizing predicted masks to {self.cfg.GWM.FLOW_RES}")
                mask_softmaxed = F.interpolate(mask_softmaxed, flow.shape[-2:], mode="bilinear", align_corners=False)

        rec_flows = self._rec_flow(sample, flow, mask_softmaxed)
        if not isinstance(rec_flows, (list, tuple)):
            rec_flows = (rec_flows,)
        k = len(rec_flows)
        loss = sum(self.criterion(flow, rec_flow) / k for rec_flow in rec_flows)
        if len(self._extra_losses):
            loss = loss + sum(self._extra_losses, 0.0) / len(self._extra_losses)
        self._extra_losses = []
        return loss

    def flow_quad(self, sample, flow, masks_softmaxed, it, **_):
        logger.debug_once(
            f"Reconstruction using quadratic. Masks shape {masks_softmaxed.shape} | "
            f"Flow shape {flow.shape} | "
            f"Grid shape {self.grid_x.shape, self.grid_y.shape}"
        )
        return flow_reconstruction.get_quad_flow(masks_softmaxed, flow, self.grid_x, self.grid_y)

    def _clipped_recon_fn(self, *args, **kwargs):
        flow = self._recon_fn(*args, **kwargs)
        flow_o = flow[:, :-2]
        flow_u = flow[:, -2:-1].clip(self.flow_u_low, self.flow_u_high)
        flow_v = flow[:, -1:].clip(self.flow_v_low, self.flow_v_high)
        return torch.cat([flow_o, flow_u, flow_v], dim=1)

    def _rec_flow(self, sample, flow, masks_softmaxed):
        it = self.it
        if self.cfg.GWM.FLOW_RES is not None and flow.shape[-2:] != self.grid_x.shape[-2:]:
            logger.debug_once(f"Generating new grid predicted masks of {flow.shape[-2:]}")
            self.grid_x, self.grid_y = utils.grid.get_meshgrid(flow.shape[-2:], self.device)
        return [self._clipped_recon_fn(sample, flow, masks_softmaxed, it)]

    def rec_flow(self, sample, flow, masks_softmaxed):
        try:
            return self._rec_flow(sample, flow, masks_softmaxed)
        except ValueError:
            return [torch.zeros_like(flow)]

    def process_flow(self, sample, flow_cuda):
        return flow_cuda

    def viz_flow(self, flow):
        return torch.stack([flow2rgb_torch(x) for x in flow])
