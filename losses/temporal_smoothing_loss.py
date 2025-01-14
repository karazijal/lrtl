import torch
import torch.nn.functional as F

import utils

from .reconstruction_loss import ReconstructionLoss

logger = utils.log.getLogger(__name__)


class TemporalSmoothingLoss(ReconstructionLoss):
    def mask_match_loss(self, mask1, mask2):
        l = (mask1 - mask2).pow(2).sum(dim=1, keepdim=True)
        return l

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        self.training = train
        flow = self.process_flow(sample, flow)
        self.it = it
        self._extra_losses = []

        # [b, f, N, 2]
        tracks = sample["track"].to(mask_softmaxed.device)  # Should be noop

        # [b, f, N, 1]
        vis = sample["vis"].to(mask_softmaxed.device)  # Should be noop

        # [b, 1, N, 2]
        coord = sample["loc"].to(mask_softmaxed.device)  # Should be noop

        # print(coord.shape, vis.shape, tracks.shape)
        b, f, N, d = tracks.shape
        K = mask_softmaxed.shape[1]
        f = vis.shape[1]

        ref_masks = F.grid_sample(mask_softmaxed, coord, align_corners=False).squeeze(-2)  # b, k, N
        # print(coord.shape, vis.shape, tracks.shape)
        b, _, D, N = tracks.shape
        K = mask_softmaxed.shape[1]
        f = vis.shape[1]

        # norm tracks
        with torch.no_grad():
            tracks = tracks.clone()
            tracks[..., 0] /= 854 - 1
            tracks[..., 1] /= 480 - 1
            tracks *= 2
            tracks -= 1

        curr_id = f // 2
        offset = 5
        loss = 0.0
        for i, j in [(0, 2), (1, 3)]:
            mask_1 = mask_softmaxed[i::4]
            mask_2 = mask_softmaxed[j::4]

            ref_mask_1 = ref_masks[i::4]
            ref_mask_2 = ref_masks[j::4]

            futr_coord = tracks[i::4, curr_id + offset].unsqueeze(1)  # [b, 1, N, 2]
            past_corrd = tracks[j::4, curr_id - offset].unsqueeze(1)

            futr_vis = vis[i::4, curr_id + offset].squeeze(-1)
            past_vis = vis[j::4, curr_id - offset].squeeze(-1)
            ref_vis_1 = vis[i::4, curr_id].squeeze(-1)  # [b, N, 1]
            ref_vis_2 = vis[j::4, curr_id].squeeze(-1)

            futr_mask = F.grid_sample(mask_2, futr_coord, align_corners=False).squeeze(-2)
            past_mask = F.grid_sample(mask_1, past_corrd, align_corners=False).squeeze(-2)

            logger.info_once(
                f"{ref_mask_1.shape=} {ref_mask_2.shape=} {futr_mask.shape=} {past_mask.shape=} {futr_vis.shape=} {past_vis.shape=} {ref_vis_1.shape=} {ref_vis_2.shape=}"
            )
            loss1_mask = futr_vis * ref_vis_1
            loss1 = (
                (self.mask_match_loss(ref_mask_1, futr_mask) * loss1_mask).sum(dim=-1)
                / loss1_mask.sum(dim=-1).clamp(1e-6)
                / 4
            )
            loss2_mask = past_vis * ref_vis_2
            loss2 = (
                (self.mask_match_loss(ref_mask_2, past_mask) * loss2_mask).sum(dim=-1)
                / loss2_mask.sum(dim=-1).clamp(1e-6)
                / 4
            )
            loss += loss1.mean() + loss2.mean()

        return loss
