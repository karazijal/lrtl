import copy
import itertools
import logging
import os
from pathlib import Path

import numpy as np
import torch.utils.data
from detectron2.config import CfgNode as CN

import utils
from datasets import FlowEvalDetectron, FlowPairDetectron, scan_dataset_2

logger = logging.getLogger("lrtl")


def scan_train_flow(folders, res, pairs, basepath):
    pair_list = [p for p in itertools.combinations(pairs, 2)]

    flow_dir = {}
    for pair in pair_list:
        p1, p2 = pair
        flowpairs = []
        for f in folders:
            path1 = basepath / f"Flows_gap{p1}" / res / f
            path2 = basepath / f"Flows_gap{p2}" / res / f

            flows1 = [p.name for p in path1.glob("*.flo")]
            flows2 = [p.name for p in path2.glob("*.flo")]

            flows1 = sorted(flows1)
            flows2 = sorted(flows2)

            intersect = list(set(flows1).intersection(flows2))
            intersect.sort()

            flowpair = np.array([[path1 / i, path2 / i] for i in intersect])
            flowpairs += [flowpair]
        flow_dir["gap_{}_{}".format(p1, p2)] = flowpairs

    # flow_dir is a dictionary, with keys indicating the flow gap, and each value is a list of sequence names,
    # each item then is an array with Nx2, N indicates the number of available pairs.
    return flow_dir


def setup_dataset(cfg=None, multi_val=False):
    dataset_str = cfg.GWM.DATASET
    resolution = cfg.GWM.RESOLUTION  # h,w
    res = ""
    with_gt = True
    pairs = [1, 2, -1, -2]
    trainval_data_dir = None
    filter_list = None
    binarise = True
    mode2017 = False

    img_res = ""
    flo_res = ""
    ano_res = ""
    anno_dir = "Annotations"

    if cfg.GWM.DATASET == "DAVIS":
        basepath = "/DAVIS2016"
        img_dir = "/DAVIS2016/JPEGImages/480p"
        gt_dir = "/DAVIS2016/Annotations/480p"

        val_flow_dir = "/DAVIS2016/Flows_gap1/1080p"
        val_seq = [
            "dog",
            "cows",
            "goat",
            "camel",
            "libby",
            "parkour",
            "soapbox",
            "blackswan",
            "bmx-trees",
            "kite-surf",
            "car-shadow",
            "breakdance",
            "dance-twirl",
            "scooter-black",
            "drift-chicane",
            "motocross-jump",
            "horsejump-high",
            "drift-straight",
            "car-roundabout",
            "paragliding-launch",
        ]
        val_data_dir = [val_flow_dir, img_dir, gt_dir]
        res = "1080p"

        img_res = "480p"
        flo_res = "1080p"
        ano_res = "480p"

    elif cfg.GWM.DATASET in ["FBMS"]:
        basepath = "/FBMS_clean"
        img_dir = "/FBMS_clean/JPEGImages/"
        gt_dir = "/FBMS_clean/Annotations/"

        val_flow_dir = "/FBMS_val/Flows_gap1/"
        val_seq = [
            "camel01",
            "cars1",
            "cars10",
            "cars4",
            "cars5",
            "cats01",
            "cats03",
            "cats06",
            "dogs01",
            "dogs02",
            "farm01",
            "giraffes01",
            "goats01",
            "horses02",
            "horses04",
            "horses05",
            "lion01",
            "marple12",
            "marple2",
            "marple4",
            "marple6",
            "marple7",
            "marple9",
            "people03",
            "people1",
            "people2",
            "rabbits02",
            "rabbits03",
            "rabbits04",
            "tennis",
        ]
        val_img_dir = "/FBMS_val/JPEGImages/"
        val_gt_dir = "/FBMS_val/Annotations/"
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        with_gt = False
        pairs = [3, 6, -3, -6]

    elif cfg.GWM.DATASET in ["STv2"]:
        basepath = "/SegTrackv2"
        img_dir = "/SegTrackv2/JPEGImages"
        gt_dir = "/SegTrackv2/Annotations"

        val_flow_dir = "/SegTrackv2/Flows_gap1/"
        val_seq = [
            "drift",
            "birdfall",
            "girl",
            "cheetah",
            "worm",
            "parachute",
            "monkeydog",
            "hummingbird",
            "soldier",
            "bmx",
            "frog",
            "penguin",
            "monkey",
            "bird_of_paradise",
        ]
        val_data_dir = [val_flow_dir, img_dir, gt_dir]

    else:
        raise ValueError("Unknown Setting/Dataset.")

    # Switching this section to pathlib, which should prevent double // errors in paths and dict keys

    root_path_str = cfg.GWM.DATA_ROOT
    logger.info(f"Found DATA_ROOT in config: {root_path_str}")
    # root_path_str = '../data'

    if root_path_str.startswith("/"):
        root_path = Path(f"/{root_path_str.lstrip('/').rstrip('/')}")
    else:
        root_path = Path(f"{root_path_str.lstrip('/').rstrip('/')}")

    logger.info(f"Loading dataset from: {root_path}")
    _basepath = basepath
    basepath = root_path / basepath.lstrip("/").rstrip("/")
    img_dir = root_path / img_dir.lstrip("/").rstrip("/")
    gt_dir = root_path / gt_dir.lstrip("/").rstrip("/")
    val_data_dir = [root_path / path.lstrip("/").rstrip("/") for path in val_data_dir]

    folders = [p.name for p in (basepath / f"Flows_gap{pairs[0]}" / res).iterdir() if p.is_dir()]
    folders = sorted(folders)

    if filter_list is not None:
        folders = [f for f in folders if f in filter_list]

    _seqs = None
    if mode2017:
        folders = [f for f in folders if f not in val_seq]
        _seqs = folders

    # flow_dir is a dictionary, with keys indicating the flow gap, and each value is a list of sequence names,
    # each item then is an array with Nx2, N indicates the number of available pairs.

    if cfg.SEQ:
        logger.info(f"SINGLE SEQUENCE MODE: {cfg.SEQ}")
        folders = [cfg.SEQ]
        val_seq = [cfg.SEQ]

    flow_dir = scan_train_flow(folders, res, pairs, basepath)
    data_dir = [flow_dir, img_dir, gt_dir, basepath]

    force1080p = ("DAVIS" not in cfg.GWM.DATASET) and "RGB_BIG" in cfg.GWM.SAMPLE_KEYS

    enable_photometric_augmentations = cfg.FLAGS.INF_TPS

    if cfg.GWM.PAIR:
        train_dataset = scan_dataset_2.build_pair(
            cfg, _basepath, img_res, flo_res, ano_res, flow_gaps=pairs[:2], ano_paths=(anno_dir,), seqs=_seqs
        )
    else:
        train_dataset = FlowPairDetectron(
            data_dir=data_dir,
            resolution=resolution,
            to_rgb=cfg.GWM.FLOW2RGB,
            size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            enable_photo_aug=enable_photometric_augmentations,
            flow_clip=cfg.GWM.FLOW_CLIP,
            norm=cfg.GWM.FLOW_NORM,
            force1080p=force1080p,
            flow_res=cfg.GWM.FLOW_RES,
            load_tracks=cfg.TRACKS,
            cfg=cfg,
        )

    val_dataset = FlowEvalDetectron(
        data_dir=val_data_dir,
        resolution=resolution,
        pair_list=pairs,
        val_seq=val_seq,
        to_rgb=cfg.GWM.FLOW2RGB,
        with_rgb=False,
        size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
        flow_clip=cfg.GWM.FLOW_CLIP,
        norm=cfg.GWM.FLOW_NORM,
        force1080p=force1080p,
        overwrite_hw=cfg.GWM.BIG_EVAL,
        binarise=binarise,
        mode2017=mode2017,
    )

    return train_dataset, val_dataset


def collate_fn(sample):
    sample = [e for s in sample for e in s]
    return torch.utils.data.default_collate(sample)


def loaders(cfg):
    train_dataset, val_dataset = setup_dataset(cfg)
    logger.info(f"Sourcing data from {val_dataset.data_dir[0]}")

    if cfg.FLAGS.DEV_DATA:
        subset = max(cfg.SOLVER.IMS_PER_BATCH * 3, 10)
        # train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))

    g = torch.Generator()
    data_generator_seed = int(torch.randint(int(1e6), (1,)).item())
    logger.info(f"Dataloaders generator seed {data_generator_seed}")
    g.manual_seed(data_generator_seed)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    if cfg.GWM.PAIR:
        batch_size = batch_size // 4
        logger.critical("PAIR MODE: Batch size divided by 4")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.DATALOADER.NUM_WORKERS > 0,
        worker_init_fn=utils.random_state.worker_init_function,
        generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=False,
        worker_init_fn=utils.random_state.worker_init_function,
        generator=g,
    )
    return train_loader, val_loader


def multi_loaders(cfg):
    train_dataset, val_datasets, train_val_datasets = setup_dataset(cfg, multi_val=True)
    logger.info(f"Sourcing multiple loaders from {len(val_datasets)}")
    logger.info(f"Sourcing data from {val_datasets[0].data_dir[0]}")

    g = torch.Generator()
    data_generator_seed = int(torch.randint(int(1e6), (1,)).item())
    logger.info(f"Dataloaders generator seed {data_generator_seed}")
    g.manual_seed(data_generator_seed)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    if cfg.GWM.PAIR:
        batch_size = batch_size // 4
        logger.critical("PAIR MODE: Batch size divided by 4")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.DATALOADER.NUM_WORKERS > 0,
        worker_init_fn=utils.random_state.worker_init_function,
        generator=g,
    )

    val_loaders = [
        (
            torch.utils.data.DataLoader(
                val_dataset,
                num_workers=0,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                collate_fn=lambda x: x,
                drop_last=False,
                persistent_workers=False,
                worker_init_fn=utils.random_state.worker_init_function,
                generator=g,
            ),
            torch.utils.data.DataLoader(
                tv_dataset,
                num_workers=0,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                shuffle=True,
                pin_memory=False,
                collate_fn=lambda x: x,
                drop_last=False,
                persistent_workers=False,
                worker_init_fn=utils.random_state.worker_init_function,
                generator=g,
            ),
        )
        for val_dataset, tv_dataset in zip(val_datasets, train_val_datasets)
    ]

    return train_loader, val_loaders


def add_gwm_config(cfg):
    cfg.GWM = CN()
    cfg.GWM.MODEL = "MASKFORMER"
    cfg.GWM.RESOLUTION = (192, 352)
    cfg.GWM.FLOW_RES = (480, 854)
    cfg.GWM.SAMPLE_KEYS = ["rgb"]
    cfg.GWM.CRITERION = "L2"
    cfg.GWM.L1_OPTIMIZE = False
    cfg.GWM.HOMOGRAPHY = "quad"
    cfg.GWM.DATASET = "DAVIS"
    cfg.GWM.PAIR = True
    cfg.GWM.DATA_ROOT = "/scratch/shared/beegfs/laurynas/datasets"
    cfg.GWM.FLOW2RGB = False
    cfg.GWM.SIMPLE_REC = False
    cfg.GWM.USE_MULT_FLOW = False
    cfg.GWM.FLOW_COLORSPACE_REC = None

    cfg.GWM.FLOW_CLIP_U_LOW = float("-inf")
    cfg.GWM.FLOW_CLIP_U_HIGH = float("inf")
    cfg.GWM.FLOW_CLIP_V_LOW = float("-inf")
    cfg.GWM.FLOW_CLIP_V_HIGH = float("inf")

    cfg.GWM.FLOW_CLIP = float("inf")
    cfg.GWM.FLOW_NORM = False

    cfg.GWM.LOSS_MULT = CN()
    cfg.GWM.LOSS_MULT.REC = 0.03
    cfg.GWM.LOSS_MULT.HEIR_W = [1.0, 1.0, 1.0, 1.0]
    cfg.GWM.LOSS_MULT.TEMP = 0.1

    
    cfg.GWM.LOSS = "svdvals-flow"
    cfg.GWM.DOF = 5


    cfg.FLAGS = CN()
    cfg.FLAGS.DEV_DATA = False
    cfg.FLAGS.KEEP_ALL = True  # Keep all checkoints
    cfg.FLAGS.ORACLE_CHECK = True  # Use oracle check to estimate max performance when grouping multiple components

    cfg.FLAGS.INF_TPS = False

    cfg.FLAGS.UNFREEZE_AT = [(4, 0), (2, 500), (1, 1000), (-1, 10000)]

    cfg.TRACKS = "co2-noshmem-multi"
    cfg.TRACK_LOSS_MULT = 0.00005
    cfg.NTRACKS = 70000
    cfg.TRACK_VIS_THRESH = 0.1  # 0.1 of the track needs to be vis
    cfg.FILTERN = 11
    cfg.CTX_LEN = 20
    cfg.CTX_MODE = "cont"
    cfg.GWM.BIG_EVAL = False

    cfg.WANDB = CN()
    cfg.WANDB.ENABLE = False

    cfg.DEBUG = False

    cfg.LOG_ID = "exp"
    cfg.LOG_FREQ = 250
    cfg.LOG_EXTRA = [5, 50, 500]
    cfg.OUTPUT_BASEDIR = "outputs"
    cfg.SKIP_TB = False
    cfg.TOTAL_ITER = 20000

    cfg.SEQ = None

    cfg.EMA = CN()
    cfg.EMA.BETA = 1.0
    cfg.EMA.UPDATE_AFTER_STEP = 1500
    cfg.EMA.UPDATE_EVERY = 10
    cfg.EMA.POWER = 2.0 / 3.0

    cfg.RANDOM_FLIP = True

    if os.environ.get("SLURM_JOB_ID", None):
        cfg.LOG_ID = os.environ.get("SLURM_JOB_NAME", cfg.LOG_ID)
        logger.info(f"Setting name {cfg.LOG_ID} based on SLURM job name")
