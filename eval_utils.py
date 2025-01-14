import functools
import os
import random
from pathlib import Path
from collections import defaultdict

from datetime import datetime

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import SpectralClustering
from tqdm.auto import tqdm

import flow_reconstruction
from utils import visualisation, log, grid
from utils.vit_extractor import ViTExtractor

from davis2017.evaluation import DAVISEvaluation

label_colors = visualisation.create_label_colormap()
logger = log.getLogger('lrtl')


def __default_font(fontsize):
    try:
        FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", fontsize)
    except OSError:
        FNT = ImageFont.truetype("dejavu/DejaVuSans.ttf", fontsize)
    return FNT


def is_slurm():
    return os.environ.get('SLURM_JOB_ID', None) and os.environ['SLURM_JOB_NAME'] not in  ['zsh', 'bash']


@functools.lru_cache(None)  # cache the result
def autosized_default_font(size_limit: float) -> ImageFont.ImageFont:
    fontsize = 1  # starting font size
    font = __default_font(fontsize)
    while font.getsize('test123')[1] < size_limit:
        fontsize += 1
        font = __default_font(fontsize)
    fontsize -= 1
    font = __default_font(fontsize)
    return font


def iou(masks, gt, thres=0.5):
    masks = (masks > thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [0, 1]))
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect / union.clip(min=1e-12)

PAL = np.array([
    [0, 0, 0],
    [255,193,7], # nice clr-blind friendly yellow
    [216, 27, 96], # nice clr-blind friendly red
    [0,77,64], # nice clr-blind friendly green
    [30, 136, 229], # nice clr-blind friendly blue
    [147,20,178], # nice clr-blind friendly purple
    [72,235,127] # nice clr-blind friendly light green
], dtype=np.uint8).reshape(-1, 3)

# Taken from DINOSAUR paper code at https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/metrics/utils.py
def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).

    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)

    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)

    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both true_mask and pred_mask assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == n_fg_points * (n_fg_points-1))
    # 2. If both true_mask and pred_mask assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def fg_adjusted_rand_index(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1 :]), dim=-1
        )

    return adjusted_rand_index(pred_mask, true_mask_only_fg)

def adjusted_rand_index_sklearn(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    from sklearn.metrics import adjusted_rand_score as adjusted_rand_score_sklearn_impl
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1).detach().cpu()
    b, n = pred_cluster_ids.shape
    if len(true_mask.shape) == 3:
        true_cluster_ids = true_mask.argmax(-1).detach().cpu()
    elif len(true_mask.shape) == 2:
        true_cluster_ids = true_mask.detach().cpu()
    assert true_cluster_ids.shape == pred_cluster_ids.shape

    true_cluster_ids = true_cluster_ids.numpy()
    pred_cluster_ids = pred_cluster_ids.numpy()
    aris = []
    for i in range(b):
        ari = adjusted_rand_score_sklearn_impl(true_cluster_ids[i], pred_cluster_ids[i])
        aris.append(ari)
    
    return torch.from_numpy(np.array(aris)).to(pred_mask.device).to(pred_mask.dtype)


def fg_adjusted_rand_index_sklearn(pred_mask, true_mask):
    from sklearn.metrics import adjusted_rand_score as adjusted_rand_score_sklearn_impl
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1).detach().cpu()
    b, n = pred_cluster_ids.shape
    if len(true_mask.shape) == 3:
        true_cluster_ids = true_mask.argmax(-1).detach().cpu()
    elif len(true_mask.shape) == 2:
        true_cluster_ids = true_mask.detach().cpu()
    assert true_cluster_ids.shape == pred_cluster_ids.shape

    true_cluster_ids = true_cluster_ids.numpy()
    pred_cluster_ids = pred_cluster_ids.numpy()
    aris = []
    for i in range(b):
        tm = true_cluster_ids[i]
        pm = pred_cluster_ids[i]
        non_bg = tm != 0
        tm = tm[non_bg]
        pm = pm[non_bg]
        ari = adjusted_rand_score_sklearn_impl(tm, pm)
        aris.append(ari)
    
    return torch.from_numpy(np.array(aris)).to(pred_mask.device).to(pred_mask.dtype)

def get_unsup_image_viz(model, cfg, sample, criterion):
    if model.training:
        model.eval()
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True, raw_sem_seg=True)
        model.train()
    else:
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True, raw_sem_seg=True)
    return get_image_vis(model, cfg, sample, preds, criterion)

def get_vis_header(header_size, image_size, header_texts, header_height=20):
    W, H = (image_size, header_height)
    header_labels = []
    font = autosized_default_font(0.8 * H)

    for text in header_texts:
        im = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(im)
        w, h = draw.textsize(text, font=font)
        draw.text(((W - w) / 2, (H - h) / 2), text, fill="black", font=font)
        header_labels.append(torch.from_numpy(np.array(im)))
    header_labels = torch.cat(header_labels, dim=1)
    ret = (torch.ones((header_height, header_size, 3)) * 255)
    ret[:, :header_labels.size(1)] = header_labels

    return ret.permute(2, 0, 1).clip(0, 255).to(torch.uint8)

def get_image_vis(model, cfg, sample, preds, criterion):
    rgb = sample['rgb'].cpu()
    masks_pred = torch.stack([x['sem_seg'] for x in preds], 0)

    with torch.no_grad():
        flow = sample['flow'].clip(-20, 20)
        flow = F.interpolate(flow, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        masks_pred = F.interpolate(masks_pred, size=rgb.shape[-2:], mode='bilinear', align_corners=False)

    masks_softmaxed = torch.softmax(masks_pred, dim=1)
    logger.info(f'Flow shape {flow.shape} masks_softmaxed shape {masks_softmaxed.shape}')
    masks_pred = masks_softmaxed
    rec_flows = criterion.flow_reconstruction(sample, criterion.process_flow(sample, flow).to(masks_softmaxed.device), masks_softmaxed)
    rec_headers = ['rec_flow']
    if len(rec_flows) > 1:
        rec_headers.append('rec_bwd_flow')

    flow = criterion.viz_flow(criterion.process_flow(sample, flow).cpu()) * 255
    rec_flows = [
        (criterion.viz_flow(rec_flow_.detach().cpu()) * 255).clip(0, 255).to(torch.uint8) for rec_flow_ in rec_flows
    ]


    gt_labels = sample['sem_seg']
    gt = F.one_hot(gt_labels, gt_labels.max().item() + 1).permute(0, 3, 1, 2)
    target_K = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
    masks = F.one_hot(masks_pred.argmax(1).cpu(), target_K).permute(0, 3, 1, 2)
    masks_each = torch.stack([masks_softmaxed, masks_softmaxed, masks_softmaxed], 2) * 255
    masks_each = einops.rearrange(F.pad(masks_each.cpu(), pad=[0, 1], value=255), 'b n c h w -> b c h (n w)')

    gt_seg = torch.einsum('b k h w, k c -> b c h w', gt, label_colors[:gt_labels.max().item() + 1])
    pred_seg = torch.einsum('b k h w, k c -> b c h w', masks, label_colors[:target_K])
   
    print(rgb.shape, flow.shape, gt_seg.shape, pred_seg.shape)
    image_viz = torch.cat([rgb, flow, gt_seg.cpu(), pred_seg.cpu(), *rec_flows], -1)
    header_text = ['rgb', 'gt_flow', 'gt_seg', 'pred_seg', *rec_headers]

    image_viz = torch.cat([image_viz, masks_each], -1)
    header_text.extend(['slot'] * masks_softmaxed.shape[1])
    image_viz = einops.rearrange(image_viz[:8], 'b c h w -> c (b h) w').detach().clip(0, 255).to(torch.uint8)

    return image_viz, header_text


def get_frame_vis(model, cfg, sample, preds):
    masks_pred = torch.stack([x['sem_seg'] for x in preds], 0)
    flow = torch.stack([x['flow'].to(model.device) for x in sample]).clip(-20, 20)

    masks_softmaxed = torch.softmax(masks_pred, dim=1)
    if cfg.GWM.SIMPLE_REC:
        mask_denom = einops.reduce(masks_softmaxed, 'b k h w -> b k 1', 'sum') + 1e-7
        means = torch.einsum('brhw, bchw -> brc', masks_softmaxed, flow) / mask_denom
        rec_flow = torch.einsum('bkhw, bkc-> bchw', masks_softmaxed, means)
    elif cfg.GWM.HOMOGRAPHY:
        rec_flow = flow_reconstruction.get_quad_flow(masks_softmaxed, flow)
    else:
        grid_x, grid_y = grid.get_meshgrid(cfg.GWM.RESOLUTION, model.device)
        rec_flow = flow_reconstruction.get_quad_flow(masks_softmaxed, flow, grid_x, grid_y)

    rgb = torch.stack([x['rgb'] for x in sample])
    flow = torch.stack([visualisation.flow2rgb_torch(x) for x in flow.cpu()]) * 255
    rec_flow = torch.stack([visualisation.flow2rgb_torch(x) for x in rec_flow.detach().cpu()]) * 255

    gt_labels = torch.stack([x['sem_seg'] for x in sample])
    gt = F.one_hot(gt_labels, gt_labels.max().item() + 1).permute(0, 3, 1, 2)

    masks = F.one_hot(masks_pred.argmax(1).cpu(), cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES).permute(0, 3, 1, 2)

    gt_seg = torch.einsum('b k h w, k c -> b c h w', gt, label_colors[:gt_labels.max().item() + 1])
    pred_seg = torch.einsum('b k h w, k c -> b c h w', masks, label_colors[:cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES])
    frame_vis = torch.cat([rgb, flow, gt_seg.cpu(), pred_seg.cpu(), rec_flow.clip(0, 255).to(torch.uint8)], -1)
    frame_vis = einops.rearrange(frame_vis, 'b c h w -> b c h w').detach().clip(0, 255).to(torch.uint8)
    return frame_vis


def is_2comp_dataset(dataset):
    if '+' in dataset:
        d = dataset.split('+')[0].strip()
    else:
        d = dataset.strip()
    logger.info_once(f"Is 2comp dataset? {d}")
    if 'DAVIS' in d and '17' in d:
        return False
    for s in ['DAVIS', 'FBMS', 'STv2']:
        if s in d:
            return True
    return d in ['DAVIS',
                 'FBMS',
                 'STv2']

def eval_unsupmf(cfg, val_loader, model, criterion, writer=None, writer_iteration=0, use_wandb=False, output_dir=None, progbar=False):
    is_2comp = is_2comp_dataset(cfg.GWM.DATASET)
    logger.info(f'Running Evaluation: {cfg.LOG_ID} {"Simple" if cfg.GWM.SIMPLE_REC else "Gradient"}:')
    logger.info(f'Model mode: {"train" if model.training else "eval"}, wandb: {use_wandb}')
    logger.info(f'Dataset: {cfg.GWM.DATASET} # components: {cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES} {is_2comp=}')

    merger = None
    if cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES > 2:
        merger = MaskMerger(cfg, model)

    print_idxs = random.sample(range(len(val_loader)), k=10)

    images_viz = []
    ious_davis_eval = defaultdict(list)
    ious = defaultdict(list)

    ious_oracle = defaultdict(list)
    ious_oracle_davis_eval = defaultdict(list)

    n = 0
    fg_ari_sum= 0
    ari_sum= 0
    # fg_ari_sum_2= 0
    # ari_sum_2= 0
    if not is_2comp:
        val_dir = Path(cfg.OUTPUT_DIR) / f'val_{writer_iteration:>06d}'
        val_dir.mkdir(exist_ok=True)

    total_time = 0
    total_time_n = 0
    for idx, sample in enumerate(tqdm(val_loader, disable=(is_slurm() and not progbar))):
        t = 1
        # sample = [e for s in sample for e in s]
        # category = [s['category'] for s in sample]
        for k in sample:
            if torch.is_tensor(sample[k]):
                logger.info_once(f"Eval sample {k} shape {sample[k].shape}")
            if k in cfg.GWM.SAMPLE_KEYS or k == 'rgb':
                sample[k] = sample[k].to(model.device)
        logger.info_once(f'Here {sample["height"]=} {sample["width"]=}')

        start_time = datetime.now()
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True)
        masks_raw = torch.stack([x['sem_seg'] for x in preds], 0)

        masks_softmaxed = torch.softmax(masks_raw, dim=1)
        logger.info_once(f'masks_softmaxed predicted shape {masks_softmaxed.shape}')

        if merger:
            masks_dict = merger(sample, masks_softmaxed)
        else:
            masks_dict = {'cos': masks_softmaxed}
        total_time += (datetime.now() - start_time).total_seconds()
        total_time_n += 1

        if writer and idx in print_idxs:
            # flow = torch.stack([x['flow'] for x in sample]).to(model.device)
            img_viz, header_text = get_image_vis(model, cfg, sample, preds, criterion)
            images_viz.append(img_viz)
        
        gt_seg = sample['sem_seg_ori'].to(model.device)
        seq = sample['category'][0][0]
        frame_name = sample['category'][1][0]
        # logger.info(f'category {seq} {frame_name}')
        if not is_2comp:
            logger.info_once(f'gt_seg shape {gt_seg.shape}')
            hw = gt_seg.shape[-2:]
            ngt = gt_seg.max().item() + 1
            npd = masks_softmaxed.shape[1]
            ncomp = max(ngt, npd)
            gt_seg_oh = F.one_hot(gt_seg, ncomp).view(gt_seg.shape[0], -1, ncomp)
            if masks_softmaxed.shape[-2:] != gt_seg_oh.shape[-2:]:
                logger.info_once(f"Upsampling predicted masks to {hw} for evaluation")
                masks_softmaxed = F.interpolate(masks_softmaxed, size=hw, mode='bilinear', align_corners=False)
            
            # logger.info(f'{masks_softmaxed.shape=}, {seq=}, {frame_name=}')

            pred_seg = masks_softmaxed.permute(0, 2, 3, 1).view(masks_softmaxed.shape[0], -1, masks_softmaxed.shape[1])
            logger.info_once(f'pred_seg shape {pred_seg.shape} gt_seg_oh shape {gt_seg_oh.shape}')

            fg_ari = fg_adjusted_rand_index(pred_seg, gt_seg_oh)
            ari = adjusted_rand_index(pred_seg, gt_seg_oh)


            logger.info_once(f'fg_ari shape {fg_ari.shape} ari shape {ari.shape}')

            ious[seq].append(torch.tensor(0.0).view(1))
            n += pred_seg.shape[0]

            fg_ari_sum += fg_ari.sum()
            ari_sum += ari.sum()
            # fg_ari_sum_2 += fg_ari_2.sum()
            # ari_sum_2 += ari_2.sum()

            mask = masks_softmaxed.argmax(1).detach().cpu()[0]
            mask_path = val_dir / seq / frame_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            mask = Image.fromarray((mask.numpy()).astype(np.uint8))
            mask.save(mask_path.with_suffix('.png'))

            continue 

        masks = masks_dict['cos']
        HW = gt_seg.shape[-2:]
        gt_seg = gt_seg.cpu()
        if HW != masks.shape[-2:]:
            logger.info_once(f"Upsampling predicted masks to {HW} for evaluation")
            masks_softmaxed_sel = F.interpolate(masks.detach().cpu(), size=HW, mode='bilinear', align_corners=False)
        else:
            masks_softmaxed_sel = masks.detach().cpu()
        masks_ = einops.rearrange(masks_softmaxed_sel, '(b t) s h w -> b t s 1 h w', t=t).detach()
        gt_seg = einops.rearrange(gt_seg, 'b h w -> b 1 h w').float()
        for i in range(masks_.size(0)):
            assert i == 0
            masks_k = F.interpolate(masks_[i], size=(1, gt_seg.shape[-2], gt_seg.shape[-1]))  # t s 1 h w
            mask_iou = iou(masks_k[:, :, 0], gt_seg[i, 0], thres=0.5)  # t s
            iou_max, slot_max = mask_iou.max(dim=1)
            if output_dir:
                # print(masks_k[:, slot_max, 0].shape, seq, frame_name)
                save_mask = masks_k[:, slot_max, 0].squeeze().detach().cpu().numpy()
                save_mask_img = Image.fromarray((save_mask * 255).astype(np.uint8))
                _output_path = output_dir / 'soft' / f"{seq}" / f"{frame_name.replace('.jpg', '.png')}"
                _output_path.parent.mkdir(exist_ok=True, parents=True)
                save_mask_img.save(_output_path)
                save_mask_img = Image.fromarray((save_mask > 0.5).astype(np.uint8) * 255)
                
                _output_path = output_dir / 'hard' / f"{seq}" / f"{frame_name.replace('.jpg', '.png')}"
                _output_path.parent.mkdir(exist_ok=True, parents=True)
                save_mask_img.save(_output_path)
            
            ious[seq].append(iou_max)
            frame_id = frame_name
            ious_davis_eval[seq].append((frame_id.strip().replace('.png', ''), iou_max))
        if output_dir:
            masks_softmaxed_scaled = F.interpolate(masks_softmaxed, size=HW, mode='bilinear', align_corners=False).detach()[0].argmax(0).cpu().numpy()
            save_mask_img = Image.fromarray((PAL[masks_softmaxed_scaled]).astype(np.uint8))
            _output_path = output_dir / 'comp' / f"{seq}" / f"{frame_name.replace('.jpg', '.png')}"
            _output_path.parent.mkdir(exist_ok=True, parents=True)
            save_mask_img.save(_output_path)
        if 'oracle' in masks_dict:
            omasks = masks_dict['oracle']
            if HW != masks.shape[-2:]:
                omasks_softmaxed_sel = F.interpolate(omasks.detach().cpu(), size=HW, mode='bilinear', align_corners=False)
            else:
                omasks_softmaxed_sel = omasks.detach().cpu()
            omasks_ = einops.rearrange(omasks_softmaxed_sel, '(b t) s h w -> b t s 1 h w', t=t).detach()
            for i in range(omasks_.size(0)):
                masks_k = F.interpolate(omasks_[i], size=(1, gt_seg.shape[-2], gt_seg.shape[-1]))  # t s 1 h w
                mask_iou = iou(masks_k[:, :, 0], gt_seg[i, 0], thres=0.5)  # t s
                iou_max, slot_max = mask_iou.max(dim=1)

                ious_oracle[seq].append(iou_max)
                frame_id = frame_name
                ious_oracle_davis_eval[seq].append((frame_id.strip().replace('.png', ''), iou_max))

    frameious = sum(ious.values(), [])
    frame_mean_iou = torch.cat(frameious).sum().item() * 100 / len(frameious)

    frameious_oracle = 0
    frame_mean_iou_oracle = 0


    if len(ious_oracle) > 0:
        frameious_oracle = sum(ious_oracle.values(), [])
        frame_mean_iou_oracle = torch.cat(frameious_oracle).sum().item() * 100 / len(frameious_oracle)
        # logger.info(f"Oracle mIoU: {frame_mean_iou_oracle:.3f} \n")
    
    if not is_2comp:
        
        davis_dir = Path(val_loader.dataset.data_dir[0]).parent

        evaluator = DAVISEvaluation(str(davis_dir), task='unsupervised-motion', gt_set='val')
        metrics_res = evaluator.evaluate(val_dir)
        Jac, Fsc = metrics_res['J'], metrics_res['F']
        mIoU = np.mean(Jac["M"])
        Fscore = np.mean(Fsc["M"])
        JandF = (mIoU + Fscore) / 2
        print(f"mIoU: {mIoU:.3f} Fscore: {Fscore:.3f} J&F: {JandF:.3f}")
        if writer:
            header = get_vis_header(images_viz[0].size(2), sample['rgb'].size(-1), header_text)
            images_viz = torch.cat(images_viz, dim=1)
            images_viz = torch.cat([header, images_viz], dim=1)
            writer.add_image('val/images', images_viz, writer_iteration)  # C H W
            if not cfg.SEQ:
                writer.add_scalar('eval/mIoU', mIoU, writer_iteration)

            if use_wandb:
                import wandb
                wandb.log({'val/images': wandb.Image(images_viz.float())}, step=writer_iteration)
                wandb.log({'eval/fg_ari': fg_ari    }, step=writer_iteration)
                wandb.log({'eval/ari': ari}, step=writer_iteration)

                wandb.log({'eval/mIoU': mIoU}, step=writer_iteration)
                wandb.log({'eval/Fscore': Fscore}, step=writer_iteration)
                wandb.log({'eval/J&F': JandF}, step=writer_iteration)
        vanilla_eval = DAVISEvaluation(str(davis_dir), task='unsupervised', gt_set='val')
        metrics_res = vanilla_eval.evaluate(val_dir)
        Jac, Fsc = metrics_res['J'], metrics_res['F']
        mIoU_vanilla = np.mean(Jac["M"])
        Fscore_vanilla = np.mean(Fsc["M"])
        JandF_vanilla = (mIoU_vanilla + Fscore_vanilla) / 2
        print(f"mIoU: {mIoU_vanilla:.3f} Fscore: {Fscore_vanilla:.3f} J&F: {JandF_vanilla:.3f}")
        if writer and use_wandb:
            wandb.log({'eval/mIoU_vanilla': mIoU_vanilla}, step=writer_iteration)
            wandb.log({'eval/Fscore_vanilla': Fscore_vanilla}, step=writer_iteration)
            wandb.log({'eval/J&F_vanilla': JandF_vanilla}, step=writer_iteration)
        return {'IoU': mIoU}
    else:
        if 'DAVIS' in cfg.GWM.DATASET.split('+')[0]:
            logger.info_once("Using DAVIS evaluator methods for evaluting IoU -- mean of mean of sequences without first frame")
            seq_scores = dict()
            for c in ious_davis_eval:
                seq_scores[c] = np.nanmean([v.item() for n, v in ious_davis_eval[c] if int(n) > 1])

            frame_mean_iou = np.nanmean(list(seq_scores.values())) * 100

            if len(ious_oracle_davis_eval) > 0:
                seq_scores_oracle = dict()
                for c in ious_oracle_davis_eval:
                    seq_scores_oracle[c] = np.nanmean([v.item() for n, v in ious_oracle_davis_eval[c] if int(n) > 1])
                frame_mean_iou_oracle = np.nanmean(list(seq_scores_oracle.values())) * 100
                # logger.info(f"Oracle mIoU: {frame_mean_iou_oracle:.3f} \n")
        else:
            seq_scores = dict()
            for c in ious:
                seq_scores[c] = torch.cat(ious[c]).sum().item() * 100 / len(ious[c])
        
                

        if writer:
            header = get_vis_header(images_viz[0].size(2), sample['rgb'].size(-1), header_text)
            images_viz = torch.cat(images_viz, dim=1)
            images_viz = torch.cat([header, images_viz], dim=1)
            writer.add_image('val/images', images_viz, writer_iteration)  # C H W
            if not cfg.SEQ:
                writer.add_scalar('eval/mIoU', frame_mean_iou, writer_iteration)
            writer.add_scalar('oracle_do_not_use/mIoU_oracle', frame_mean_iou_oracle, writer_iteration)
            for c in seq_scores:
                writer.add_scalar(f'xtra/mIoU_{c}', seq_scores[c], writer_iteration)

            

            if use_wandb:
                import wandb
                if not cfg.SEQ:
                    wandb.log({'eval/mIoU': frame_mean_iou}, step=writer_iteration)
                wandb.log({'val/images': wandb.Image(images_viz.float())}, step=writer_iteration)
                wandb.log({'oracle_do_not_use/mIoU_oracle': frame_mean_iou_oracle}, step=writer_iteration)
                if not is_2comp_dataset(cfg.GWM.DATASET):
                    wandb.log({'eval/fg_ari': fg_ari    }, step=writer_iteration)
                    wandb.log({'eval/ari': ari}, step=writer_iteration)
                    # wandb.log({'eval/fg_ari_2': fg_ari_2}, step=writer_iteration)
                    # wandb.log({'eval/ari_2': ari_2}, step=writer_iteration)
                for c in seq_scores:
                    wandb.log({f'xtra/mIoU_{c}': seq_scores[c]}, step=writer_iteration)
    
        logger.info(f"mIoU: {frame_mean_iou:.3f} \n") 
        inf_speed = total_time / total_time_n
        logger.info(f"Inference speed {inf_speed} Hz")
        return {'IoU': frame_mean_iou}


class MaskMerger:
    def __init__(self, cfg, model, merger_model="dino_vits8"):
        self.extractor = ViTExtractor(model_type=merger_model, device=model.device)
        self.out_dim = 384

        self.mu = torch.tensor(self.extractor.mean).to(model.device).view(1, -1, 1, 1)
        self.sigma = torch.tensor(self.extractor.std).to(model.device).view(1, -1, 1, 1)
        self.start_idx = 0
        self.do_oracle = cfg.FLAGS.ORACLE_CHECK
        self.big_eval = cfg.GWM.BIG_EVAL 

    def get_feats(self, batch):
        with torch.no_grad():
            feat = self.extractor.extract_descriptors(batch, facet='key', layer=11, bin=False)
            feat = feat.reshape(feat.size(0), *self.extractor.num_patches, -1).permute(0, 3, 1, 2)
            return F.interpolate(feat, batch.shape[-2:], mode='bilinear')

    def spectral(self, A):
        clustering = SpectralClustering(n_clusters=2,
                                        affinity='precomputed',
                                        random_state=0).fit(A.detach().cpu().numpy())
        return np.arange(A.shape[-1])[clustering.labels_ == 0], np.arange(A.shape[-1])[clustering.labels_ == 1]

    def cos_merge(self, basis, masks):
        basis = basis / torch.linalg.vector_norm(basis, dim=-1, keepdim=True).clamp(min=1e-6)
        A = torch.einsum('brc, blc -> brl', basis, basis)[0].clamp(min=1e-6)
        inda, indb = self.spectral(A)
        return torch.stack([masks[:, inda].sum(1),
                            masks[:, indb].sum(1)], 1)

    def oracle_merge(self, masks, gt_masks):
        assert masks.shape[0] == gt_masks.shape[0] == 1
        t=1
        splits = None, None
        best_iou_max = -1
        for inda, indb in group_2_views(masks.shape[1]):
            merged_mask = torch.stack([masks[:, inda].sum(1), masks[:, indb].sum(1)], dim=1)
            masks_ = einops.rearrange(merged_mask, '(b t) s h w -> b t s 1 h w', t=t).detach()
            gt_seg = einops.rearrange(gt_masks, 'b h w -> b 1 h w').float()
            for i in range(masks_.size(0)):
                masks_k = F.interpolate(masks_[i], size=(1, gt_seg.shape[-2], gt_seg.shape[-1]))  # t s 1 h w
                mask_iou = iou(masks_k[:, :, 0], gt_seg[i, 0], thres=0.5)  # t s
                iou_max, slot_max = mask_iou.max(dim=1)
                if iou_max > best_iou_max:
                    best_iou_max = iou_max
                    splits = inda, indb
        return torch.stack([masks[:, splits[0]].sum(1),
                            masks[:, splits[1]].sum(1)], 1)
    

    def __call__(self, sample, masks_softmaxed):
        with torch.no_grad():
            masks_softmaxed = masks_softmaxed[:, self.start_idx:]
            basis_mask = masks_softmaxed
            batch = sample['rgb'] / 255.0
            if self.big_eval:
                batch = F.interpolate(batch, size=(240, 424), mode='bilinear', align_corners=False)
                basis_mask = F.interpolate(masks_softmaxed, size=batch.shape[-2:], mode='bilinear', align_corners=False)
            batch = batch.to(masks_softmaxed.device)
            logger.info_once(f'Merger {masks_softmaxed.shape=} {batch.shape=}')
            features = self.get_feats((batch - self.mu) / self.sigma)
            basis = torch.einsum('brhw, bchw -> brc', basis_mask, features)
            basis /= einops.reduce(basis_mask, 'b r h w -> b r 1', 'sum').clamp_min(1e-12)
            ret = {
                'cos': self.cos_merge(basis, masks_softmaxed),
            }
            if self.do_oracle:
                gt_masks = sample['sem_seg_ori'].cpu()
                masks_softmaxed_res = F.interpolate(masks_softmaxed.detach(), size=gt_masks.shape[-2:], mode='bilinear', align_corners=False).cpu()
                ret['oracle'] = self.oracle_merge(masks_softmaxed_res, gt_masks)
            return ret


import itertools

def group_2_views(n):
    itr = list(range(n))
    s = set(range(n))
    for c in itertools.chain.from_iterable(itertools.combinations(itr, r) for r in range(1, n // 2 + 1)):
        yield sorted(c), list(sorted(s - set(c)))

def generate_splits(tensor):
    for inda, indb in group_2_views(tensor.shape[1]):
        yield torch.stack([tensor[:, inda].sum(1), tensor[:, indb].sum(1)], dim=1)

