## Learning segmentation from point trajectories
#### [Laurynas Karazija*](https://karazijal.github.io), [Iro Laina](http://campar.in.tum.de/Main/IroLaina), [Christian Rupprecht](https://chrirupp.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)
### [![ProjectPage](https://img.shields.io/badge/-Project%20Page-magenta.svg?style=for-the-badge&color=white&labelColor=magenta)](https://www.robots.ox.ac.uk/~vgg/research/lrtl/) [![Conference](https://img.shields.io/badge/NeurIPS%20Spotlight-2024-purple.svg?style=for-the-badge&color=f1e3ff&labelColor=purple)](https://neurips.cc/virtual/2024/poster/93186)    [![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg?style=for-the-badge&logo=arXiv)](TODO)



### Abstract:
<sup> We consider the problem of segmenting objects in videos based on their motion and no other forms of supervision. Prior work has often approached this problem by using the principle of common fate, namely the fact that the motion of points that belong to the same object is strongly correlated. However, most authors have only considered instantaneous motion from optical flow. In this work, we present a way to train a segmentation network using long-term point trajectories as a supervisory signal to complement optical flow. The key difficulty is that long-term motion, unlike instantaneous motion, is difficult to model -- any parametric approximation is unlikely to capture complex motion patterns over long periods of time. We instead draw inspiration from subspace clustering approaches, proposing a loss function that seeks to group the trajectories into low-rank matrices where the motion of object points can be approximately explained as a linear combination of other point tracks. Our method outperforms the prior art on motion-based segmentation, which shows the utility of long-term motion and the effectiveness of our formulation. </sup>

### Getting Started

#### Requirements

The following packages are required to run the code:
 - cv2
 - numpy    
 - torch==2.0.1
 - torchvision==0.15.2
 - einops
 - timm
 - wandb
 - tqdm
 - scikit-learn
 - scipy
 - PIL
 - detectron2

see [`environment.yaml`](environment.yml) for precise versions and full list of dependencies and environment state.


#### Data Preparation

Datasets should be placed under `data/<dataset_name>`, e.g. `data/DAVIS2016`.
For video segmentation we follow the dataset preparation steps of [MotionGrouping](https://github.com/charigyang/motiongrouping) including obtaining optical flow.
For trajectoies, we use CotrackerV2. To generate trajectories run the following commands:
```bash
# DAVIS
python extract_trajectories.py data/DAVIS2016/ data/DAVIS2016/Tracks/cotrackerv2_rel_stride4_aux2 --grid_step 1 --height 480 --width 854 --max_frames 100 --grid_stride 4 --precheck
# SegTrackV2
python extract_trajectories.py data/SegTrackv2/ data/SegTrackv2/Tracks/cotrackerv2_rel_stride4_aux2 --grid_step 1 --height 480 --width 854 --grid_stride 4 --max_frames 100 --seq-search-path JPEGImages --precheck
# FBMS
python extract_trajectories.py data/FBMS_clean/ data/FBMS_clean/Tracks/ --grid_step 1 --height 480 --width 854 --grid_stride 4 --max_frames 100 --seq-search-path JPEGImages --precheck
```

Note that calculating trajectories will take a long time and requires a lot of memory due to tracking very many points (we observed that this lead to more accurate trajectories with cotracker). We made use of SLURM arrays to distribute the workload across many GPUs. We used machines with at least 64 GB of RAM and 48 GB of GPU memory. The scrip has additional options and functionality to resume, checkpoint, and skip already processed sequences. Also there are options for debugging. 


### Running
#### Training

Experiments are controlled through a mix of config files and command line arguments. See config files and [`src/config.py`](src/config.py) for a list of all available options.

```bash
python main.py GWM.DATASET DAVIS LOG_ID davis_training
```
for training a model on davis dataset.


### Checkpoints
We provide trained checkpoints for main experiments in the paper. These can be downloaded from the following links:
 - [DAVIS](https://www.robots.ox.ac.uk/~vgg/research/lrtl/checkpoints/davis_alt_best.pth)
 - [SegTrackV2](https://www.robots.ox.ac.uk/~vgg/research/lrtl/checkpoints/stv2_best.pth)
 - [FBMS](https://www.robots.ox.ac.uk/~vgg/research/lrtl/checkpoints/fbms_best.pth)


### Acknowledgements

This repository builds on [MaskFormer](https://github.com/facebookresearch/MaskFormer), [MotionGrouping](https://github.com/charigyang/motiongrouping), [guess-what-moves](https://github.com/karazijal/guess-what-moves), and [dino-vit-features](https://github.com/ShirAmir/dino-vit-features).

### Citation   
```
@inproceedings{karazija24learning,
  title={Learning segmentation from point trajectories},
  author={Karazija, Laurynas and Laina, Iro and Rupprecht, Christian and Vedaldi, Andrea},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```   