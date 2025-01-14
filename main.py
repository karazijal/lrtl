import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import gc
import os
import sys
import time
import warnings
from argparse import ArgumentParser

import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
warnings.filterwarnings("ignore", category=UserWarning, message=r".*does not have a deterministic implementation.*")


import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import PeriodicCheckpointer
from detectron2.modeling import build_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import ema
import losses
import utils
from eval_utils import eval_unsupmf, get_unsup_image_viz, get_vis_header
from mask_former_trainer import Trainer, setup
from unet import UNet

logger = utils.log.getLogger("lrtl")


def is_slurm():
    return os.environ.get('SLURM_JOB_ID', None) and os.environ['SLURM_JOB_NAME'] not in  ['zsh', 'bash']


def freeze(module, set=False):
    for param in module.parameters():
        param.requires_grad = set


def main(args):
    cfg, should_resume_checkpoint = setup(args)
    logger.info(f"Called as {' '.join(sys.argv)}")
    logger.info(f"Output dir {cfg.OUTPUT_DIR}")

    random_state = utils.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))
    random_state.seed_everything()
    torch.backends.cudnn.benchmark = False
    utils.log.checkpoint_code(cfg.OUTPUT_DIR)

    if not cfg.SKIP_TB:
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        writer = None

    # initialize model
    if cfg.MODEL.META_ARCHITECTURE == "UNET":
        model = UNet(cfg)
        model = model.to(torch.device(cfg.MODEL.DEVICE))
    else:
        # model = Trainer.build_model(cfg)
        model = build_model(cfg)  # this avoids logging... for now
    optimizer = Trainer.build_optimizer(cfg, model)
    scheduler = Trainer.build_lr_scheduler(cfg, optimizer)

    logger.info(f"Optimiser is {type(optimizer)}")

    ema_kwargs = {}
    ema_model = None
    if cfg.EMA.BETA < 1.0:
        ema_model = ema.EMA(
            model,
            beta=cfg.EMA.BETA,
            update_after_step=cfg.EMA.UPDATE_AFTER_STEP,
            update_every=cfg.EMA.UPDATE_EVERY,
            power=cfg.EMA.POWER,
            include_online_model=False,
        )
        ema_kwargs = {"ema_model": ema_model}
        logger.info(
            f"EMA model with beta {cfg.EMA.BETA} and update_every {cfg.EMA.UPDATE_EVERY} and update_after_step {cfg.EMA.UPDATE_AFTER_STEP}"
        )

    checkpointer = DetectionCheckpointer(
        model,
        save_dir=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
        random_state=random_state,
        optimizer=optimizer,
        scheduler=scheduler,
        **ema_kwargs,
    )
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer=checkpointer,
        #  period=cfg.SOLVER.CHECKPOINT_PERIOD,
        period=cfg.LOG_FREQ,
        max_iter=cfg.SOLVER.MAX_ITER,
        max_to_keep=None if cfg.FLAGS.KEEP_ALL else 5,
        file_prefix="checkpoint",
    )
    checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=should_resume_checkpoint)
    iteration = 0 if not should_resume_checkpoint else checkpoint["iteration"]

    train_loader, val_loader = config.loaders(cfg)

    # overfit single batch for debug
    # sample = next(iter(loader))

    criterion = losses.build_criterion(cfg, model)

    if args.eval_only:
        if len(val_loader.dataset) == 0:
            logger.error("Training dataset: empty")
            sys.exit(0)
        model.eval()
        iou = eval_unsupmf(
            cfg=cfg, val_loader=val_loader, model=model, criterion=criterion, writer=writer, writer_iteration=iteration
        )
        logger.info(f"Results: iteration: {iteration} IOU = {iou}")
        return
    if len(train_loader.dataset) == 0:
        logger.error("Training dataset: empty")
        sys.exit(0)

    logger.info(
        f"Start of training: dataset {cfg.GWM.DATASET},"
        f" train {len(train_loader.dataset)}, val {len(val_loader.dataset)},"
        f" device {model.device}, keys {cfg.GWM.SAMPLE_KEYS}, "
        f"multiple flows {cfg.GWM.USE_MULT_FLOW}"
    )

    best_metric = 0
    timestart = time.time()
    dilate_kernel = torch.ones((2, 2), device=model.device)

    total_iter = cfg.TOTAL_ITER if cfg.TOTAL_ITER else cfg.SOLVER.MAX_ITER  # early stop
    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and tqdm(
        initial=iteration, total=total_iter, disable=is_slurm()
    ) as pbar:
        while iteration < total_iter:
            for sample in train_loader:

                # if cfg.MODEL.META_ARCHITECTURE != 'UNET' and cfg.FLAGS.UNFREEZE_AT:
                #     if hasattr(model.backbone, 'frozen_stages'):
                #         assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, f"MODEL initial parameters forced frozen"
                #         stages = [s for s, m in cfg.FLAGS.UNFREEZE_AT]
                #         milest = [m for s, m in cfg.FLAGS.UNFREEZE_AT]
                #         pos = bisect.bisect_right(milest, iteration) - 1
                #         if pos >= 0:
                #             curr_setting = model.backbone.frozen_stages
                #             if curr_setting != stages[pos]:
                #                 logger.info(f"Updating backbone freezing stages from {curr_setting} to {stages[pos]}")
                #                 model.backbone.frozen_stages = stages[pos]
                #                 model.train()
                #     else:
                #         assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, f"MODEL initial parameters forced frozen"
                #         stages = [s for s, m in cfg.FLAGS.UNFREEZE_AT]
                #         milest = [m for s, m in cfg.FLAGS.UNFREEZE_AT]
                #         pos = bisect.bisect_right(milest, iteration) - 1
                #         freeze(model, set=False)
                #         freeze(model.sem_seg_head.predictor, set=True)
                #         if pos >= 0:
                #             stage = stages[pos]
                #             if stage <= 2:
                #                 freeze(model.sem_seg_head, set=True)
                #             if stage <= 1:
                #                 freeze(model.backbone, set=True)
                #         model.train()

                # else:
                #     logger.debug_once(f'Unfreezing disabled schedule: {cfg.FLAGS.UNFREEZE_AT}')

                # logger.info_once(f"Sample {len(sample)} of {len(sample[0])} items")
                # sample = [e for s in sample for e in s]
                logged = logger.info_once(f"Sample keys {[k for k in sample]}")
                excl_keys = []
                excl_sample = {}
                for k in sample:
                    if torch.is_tensor(sample[k]):
                        if any(n in k for n in ["track", "vis", "rgb", "flow", "loc", *cfg.GWM.SAMPLE_KEYS]):
                            sample[k] = sample[k].to(model.device, non_blocking=True)
                        if not logged:
                            logger.info_once(f"Sample key {k} shape {sample[k].shape}")
                    elif not logged:
                        logger.info_once(f"Sample key {k} {type(sample[k])}")
                    if "sem_seg" in k:
                        excl_keys.append(k)
                        excl_sample[k] = sample[k]
                for k in excl_keys:
                    del sample[k]
                    logger.info_once(f"Dropping {k} during training")

                raw_sem_seg = False
                if cfg.GWM.FLOW_RES is not None:
                    # flow_key = 'flow_big'
                    raw_sem_seg = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME == "MegaBigPixelDecoder"

                # flow = torch.stack([x[flow_key].to(model.device) for x in sample]).clip(-20, 20)
                flow = sample["flow"].clip(-20, 20)
                logger.debug_once(f"flow shape: {flow.shape}")
                preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True, raw_sem_seg=raw_sem_seg)
                masks_raw = torch.stack([x["sem_seg"] for x in preds], 0)
                logger.info_once(f"Output Mask shape: {masks_raw.shape}")
                masks_softmaxed_list = [torch.softmax(masks_raw, dim=1)]

                total_losses = []
                log_dicts = []
                for mask_idx, masks_softmaxed in enumerate(masks_softmaxed_list):

                    loss, log_dict = criterion(sample, flow, masks_softmaxed, iteration)

                    if cfg.GWM.USE_MULT_FLOW:
                        flow2 = sample["flow2"].clip(-20, 20)
                        other_loss, other_log_dict = criterion(sample, flow2, masks_softmaxed, iteration)
                        loss = loss / 2 + other_loss / 2
                        for k, v in other_log_dict.items():
                            log_dict[k] = other_log_dict[k] / 2 + v / 2
                        del flow2, other_loss, other_log_dict

                    total_losses.append(loss)
                    log_dicts.append(log_dict)

                loss_ws = cfg.GWM.LOSS_MULT.HEIR_W
                total_w = float(sum(loss_ws[: len(total_losses)]))
                log_dict = {}
                if len(total_losses) == 1:
                    # print('no loss reweighting')
                    log_dict = log_dicts[0]
                    loss = total_losses[0]
                else:
                    print("loss reweithgting")
                    loss = 0
                    for i, (tl, w, ld) in enumerate(zip(total_losses, loss_ws, log_dicts)):
                        for k, v in ld.items():
                            log_dict[f"{k}_{i}"] = v * w / total_w
                        loss += tl * w / total_w

                train_log_dict = {f"train/{k}": v for k, v in log_dict.items()}
                train_log_dict["train/learning_rate"] = optimizer.param_groups[-1]["lr"]
                train_log_dict["train/loss_total"] = loss.item()

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix(**{k.replace("loss_", ""): v for k, v in log_dict.items()})

                if ema_model:
                    ema_model.update()

                del log_dict, flow, preds, masks_raw, masks_softmaxed_list, total_losses
                pbar.update()

                # Sanity check for RNG state
                if (iteration + 1) % 1000 == 0 or iteration + 1 in {1, 50}:
                    logger.info(
                        f"Iteration {iteration + 1}. RNG outputs {utils.random_state.get_randstate_magic_numbers(model.device)}"
                    )

                if cfg.DEBUG or (iteration + 1) % 100 == 0:
                    logger.info(
                        f"Iteration: {iteration + 1}, time: {time.time() - timestart:.01f}s, loss: {loss.item():.02f}."
                    )

                    for k, v in train_log_dict.items():
                        if writer:
                            writer.add_scalar(k, v, iteration + 1)

                    if cfg.WANDB.ENABLE:
                        wandb.log(train_log_dict, step=iteration + 1)

                with torch.no_grad():
                    if (iteration + 1) % cfg.LOG_FREQ == 0 or (iteration + 1) in cfg.LOG_EXTRA:
                        model.eval()
                        if writer or cfg.WANDB.ENABLE:
                            # flow = torch.stack([x['flow'].to(model.device) for x in sample]).clip(-20, 20)
                            sample.update(excl_sample)

                            image_viz, header_text = get_unsup_image_viz(model, cfg, sample, criterion)
                            header = get_vis_header(image_viz.size(2), sample["rgb"].size(-1), header_text)
                            image_viz = torch.cat([header, image_viz], dim=1)
                            if writer:
                                writer.add_image("train/images", image_viz, iteration + 1)
                            if cfg.WANDB.ENABLE:
                                # image_viz = get_unsup_image_viz(model, cfg, sample, criterion)
                                wandb.log({"train/viz": wandb.Image(image_viz.float())}, step=iteration + 1)
                            del image_viz, header_text
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                        eval_model = model if not ema_model else ema_model.ema_model
                        eval_model.eval()
                        metric_dict = eval_unsupmf(
                            cfg=cfg,
                            val_loader=val_loader,
                            model=eval_model,
                            criterion=criterion,
                            writer=writer,
                            writer_iteration=iteration + 1,
                            use_wandb=cfg.WANDB.ENABLE,
                        )
                        metric_name, metric_value = list(metric_dict.items())[0]

                        if metric_value > 0.0:
                            if cfg.SOLVER.CHECKPOINT_PERIOD and metric_value > best_metric:
                                best_metric = metric_value
                                if not args.wandb_sweep_mode:
                                    checkpointer.save(
                                        name="checkpoint_best",
                                        iteration=iteration + 1,
                                        loss=loss,
                                        **{metric_name: best_metric},
                                    )
                                logger.info(
                                    f"New best {metric_name} {best_metric:.02f} after iteration {iteration + 1}"
                                )

                            if not cfg.SEQ:  # Do not report best results for single sequence
                                if cfg.WANDB.ENABLE:
                                    wandb.log(
                                        {f"eval/{metric_name}_best": best_metric}, step=iteration + 1, commit=True
                                    )
                                if writer:
                                    writer.add_scalar(f"eval/{metric_name}_best", best_metric, iteration + 1)

                        model.train()
                        logger.info(f"GC: {gc.collect()}")
                periodic_checkpointer.step(iteration=iteration + 1, loss=loss)

                iteration += 1
                timestart = time.time()
                del sample


def get_argparse_args():
    parser = ArgumentParser()
    parser.add_argument("--unique", default=False, action="store_true")
    parser.add_argument("--dev", default=False, action="store_true")
    parser.add_argument("--use_wandb", dest="wandb_sweep_mode", action="store_true")  # for sweep
    parser.add_argument("--config-file", type=str, default="configs/maskformer/maskformer_R50_bs16_160k_dino.yaml")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs="*",
    )
    return parser


if __name__ == "__main__":
    args = get_argparse_args().parse_args()
    main(args)
