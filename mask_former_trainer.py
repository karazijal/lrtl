# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
import wandb
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import (LazyConfig, _highlight, _try_get_key,
                                        collect_env_info, seed_all_rng)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

import utils
# MaskFormer
from config import add_gwm_config

logger = logging.getLogger("lrtl")


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        elif optimizer_type == "RMSProp":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.RMSprop)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def check_and_find_checkpoint(str_path):
    p = Path(str_path)
    if p.is_dir():
        last_checkpoint_file = p / "last_checkpoint"
        if last_checkpoint_file.exists():
            with last_checkpoint_file.open("r") as inf:
                last_checkpoint = inf.read().strip()
            checkpoint = p / last_checkpoint
            if checkpoint.exists():
                return checkpoint
        checkpoints = sorted(p.glob("*.pth"), key=lambda x: x.stat().st_mtime)
        if len(checkpoints) > 0:
            logger.info(f"Found {len(checkpoints)} checkpoints in {p}. Using the latest one.")
            return checkpoints[-1]
    elif p.is_file():
        return p
    return None


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    # logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    # logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        # logger.info(
        #     "Contents of args.config_file={}:\n{}".format(
        #         args.config_file,
        #         _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
        #     )
        # )
        pass

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            # logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False)


def setup(args, no_freeze=False):
    wandb_inited = False
    will_auto_resume = False
    # load cfgs
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    from mask_former import add_mask_former_config

    add_mask_former_config(cfg)

    add_gwm_config(cfg)

    cfg.merge_from_file(args.config_file)

    # if 'vggdev11' in os.environ.get('HOSTNAME', 'wrong') and 'DAVIS' in cfg.GWM.DATASET:
    #     cfg.GWM.DATA_ROOT = '/scratch/local/ssd/laurynas/datasets'
    #     print(f"Setting DATA_ROOT to {cfg.GWM.DATA_ROOT}")

    cfg.merge_from_list(args.opts)

    assert cfg.GWM.MODEL == "MASKFORMER", "Unknown Model: {cfg.GWM.MODEL}. Exiting.."

    # setup output dir
    datestring = utils.log.get_datestring_for_the_run()
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_BASEDIR, cfg.LOG_ID)
    if args.unique:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datestring)  # Force unique output dir

    if args.dev:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_BASEDIR, "dev", datestring)
        cfg.LOG_ID = "dev"
        cfg.WANDB.ENABLE = False
        cfg.FLAGS.DEV_DATA = True

    # Set up logging
    rank = comm.get_rank()
    main_log = f"{cfg.OUTPUT_DIR}/main.log"
    if args.eval_only:
        main_log = f"{cfg.OUTPUT_DIR}/eval_{datestring}.log"
    logger = setup_logger(output=main_log, distributed_rank=rank, name="lrtl")

    checkpoint_dir = Path(f"{cfg.OUTPUT_DIR}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # ensure path and checkpoint dir exists
    checkpoint = check_and_find_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        logger.info(f"Found checkpoint {checkpoint}.")
        config_path = Path(f"{cfg.OUTPUT_DIR}/config.yaml")
        if config_path.exists():
            logger.info(f"Found resumption config: {config_path}")
            cfg.merge_from_file(config_path)
        will_auto_resume = True

    # Overwrite again if this was loaded using a checkpoint
    if args.eval_only:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "eval", datestring)  # Make unique eval directory
        logger.info(f"Eval mode: updating output_dir to {cfg.OUTPUT_DIR}")

    cfg.WANDB.ENABLE = (cfg.WANDB.ENABLE or args.wandb_sweep_mode) and not args.eval_only
    if cfg.WANDB.ENABLE and rank == 0:
        if PathManager.isfile("wandb.yaml"):
            wandb_cfg = CfgNode.load_yaml_with_base("wandb.yaml", allow_unsafe=False)
            project = wandb_cfg["PROJECT"]
            entity = wandb_cfg["USER"]
        else:
            logger.error(f"W&B config file wandb.yaml does not exist!")
            project = "gwm"
            entity = "gwm"

        wandb_basedir = Path(cfg.OUTPUT_DIR)

        name = f"{cfg.LOG_ID}"
        tags = []

        if args.unique:
            name += f"/{datestring}"
        notes = None
        if os.environ.get("SLURM_JOBID", None):
            notes = f"SLURM_JOBID: {os.environ.get('SLURM_JOBID', None)}"

        wandb.init(
            project=project,
            entity=entity,
            dir=wandb_basedir,
            name=name,
            tags=tags,
            notes=notes,
            resume=will_auto_resume,
            config=cfg,
        )
        wandb_inited = True

    default_setup(cfg, args)
    if not no_freeze:
        cfg.freeze()
    if rank == 0:
        with open(f"{cfg.OUTPUT_DIR}/args.json", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    return cfg, will_auto_resume
