
from .reconstruction_loss import ReconstructionLoss
from .track_svdval import TrackSVDValsSampled
from .temporal_smoothing_loss import TemporalSmoothingLoss
import torch


class CriterionDict:
    def __init__(self, dict):
        self.criterions = dict
        assert "reconstruction" in self.criterions, "Reconstruction criterion must be present in the dict"

    def __call__(self, sample, flow, masks_softmaxed, iteration, train=True, prefix=""):
        loss = torch.tensor(0.0, device=masks_softmaxed.device, dtype=masks_softmaxed.dtype)
        log_dict = {}
        for name_i, (criterion_i, loss_multiplier_i, anneal_fn_i) in self.criterions.items():
            loss_i = (
                loss_multiplier_i
                * anneal_fn_i(iteration)
                * criterion_i(sample, flow, masks_softmaxed, iteration, train=train)
            )
            loss += loss_i
            log_dict[f"loss_{name_i}"] = loss_i.item()

        log_dict["loss_total"] = loss.item()
        return loss, log_dict

    def flow_reconstruction(self, sample, flow, masks_softmaxed):
        return self.criterions["reconstruction"][0].rec_flow(sample, flow, masks_softmaxed)

    def can_reconstruct_flow(self):
        return "reconstruction" in self.criterions

    def process_flow(self, sample, flow):
        return self.criterions["reconstruction"][0].process_flow(sample, flow)

    def viz_flow(self, flow):
        return self.criterions["reconstruction"][0].viz_flow(flow)


class LinearDecayFn:
    def __init__(self, start_factor, end_factor, end_iter, start_iter=0):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.end_iter = end_iter
        self.start_iter = start_iter
    
    def __call__(self, iteration):
        if iteration >= self.end_iter:
            return self.end_factor
        if iteration < self.start_iter:
            return self.start_factor
        return self.start_factor + (self.end_factor - self.start_factor) * (iteration - self.start_iter) / (self.end_iter - self.start_iter)


def build_criterion(cfg, model):
    total_iters = cfg.SOLVER.MAX_ITER

    if cfg.GWM.LOSS == "gwm":
        criterions = {"reconstruction": (ReconstructionLoss(cfg, model), cfg.GWM.LOSS_MULT.REC, lambda x: 1)}
    elif cfg.GWM.LOSS == 'svdvals-flow':
        criterions = {
            "reconstruction": (ReconstructionLoss(cfg, model), cfg.GWM.LOSS_MULT.REC, lambda x: 1),
            "svdvals": (TrackSVDValsSampled(cfg, model, double=False, norm='none'), cfg.TRACK_LOSS_MULT, lambda x: 1)
        }
    else:
        raise NotImplementedError(f"Loss {cfg.GWM.LOSS} not implemented")

    if cfg.GWM.LOSS_MULT.TEMP > 0.:
        criterions["temp"] = (TemporalSmoothingLoss(cfg, model), cfg.GWM.LOSS_MULT.TEMP, LinearDecayFn(0.001, 1, total_iters))

    return CriterionDict(criterions)
