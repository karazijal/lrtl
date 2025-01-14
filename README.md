## Learning Segmentation from Point Trajectories
#### [Laurynas Karazija*](https://karazijal.github.io), [Iro Laina](http://campar.in.tum.de/Main/IroLaina), [Christian Rupprecht](https://chrirupp.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)
### [![ProjectPage](https://img.shields.io/badge/-Project%20Page-magenta.svg?style=for-the-badge&color=white&labelColor=magenta)](https://www.robots.ox.ac.uk/~vgg/research/lrtl/) [![Conference](https://img.shields.io/badge/NeurIPS%20Spotlight-2024-purple.svg?style=for-the-badge&color=f1e3ff&labelColor=purple)](https://neurips.cc/virtual/2024/poster/93186)    [![arXiv](https://img.shields.io/badge/arXiv-2205.07844-b31b1b.svg?style=for-the-badge&logo=arXiv)](https://arxiv.org/abs/2205.07844)



### Abstract:
<sup> We consider the problem of segmenting objects in videos based on their motion and no other forms of supervision. Prior work has often approached this problem by using the principle of common fate, namely the fact that the motion of points that belong to the same object is strongly correlated. However, most authors have only considered instantaneous motion from optical flow. In this work, we present a way to train a segmentation network using long-term point trajectories as a supervisory signal to complement optical flow. The key difficulty is that long-term motion, unlike instantaneous motion, is difficult to model -- any parametric approximation is unlikely to capture complex motion patterns over long periods of time. We instead draw inspiration from subspace clustering approaches, proposing a loss function that seeks to group the trajectories into low-rank matrices where the motion of object points can be approximately explained as a linear combination of other point tracks. Our method outperforms the prior art on motion-based segmentation, which shows the utility of long-term motion and the effectiveness of our formulation. </sup>

### Getting Started

#### Requirements

Create and name a conda environment of your choosing, e.g. `gwm`:
```bash
conda create -n gwm python=3.8
conda activate gwm
```
then install the requirements using this one liner:
```bash
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch && \
conda install -y kornia jupyter tensorboard timm einops scikit-learn scikit-image openexr-python tqdm gcc_linux-64=11 gxx_linux-64=11 fontconfig -c conda-forge && \
yes | pip install cvbase opencv-python wandb && \
yes | python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

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

Note that calculating trajectories will take a long time and requires a lot of memory due to tracking very many points (we observed that this lead to more accurate trajectories with cotracker). We made use of SLURM arrays to distribute the workload accross many GPUs. We used machines with at least 64 GB of RAM and 48 GB of GPU memory. The scrip has additional options and functionality to resume, checkpoint, and skip already processed sequences. Also there are options for debugging. 


### Running

#### Training

Experiments are controlled through a mix of config files and command line arguments. See config files and [`src/config.py`](src/config.py) for a list of all available options.

```bash
main.py MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 4 SOLVER.IMS_PER_BATCH 8 GWM.DATASET FBMS GWM.LOSS svdvals-flow GWM.TRACKS co2-noshmem-multi GWM.DOF 5 CTX_LEN 20 TRACK_VIS_THRESH 0.1 TRACK_LOSS_MULT 0.00005 NTRACKS 70000 LOG_EXTRA [250] GWM.RESOLUTION (192,352) FILTERN 11 LOG_FREQ 1000 EMA.BETA 0.99 RANDOM_FLIP True GWM.PAIR True GWM.LOSS_MULT.TEMP 0.1
```
Run the above commands in [`src`](src) folder.

#### Evaluation

Evaluation scripts are provided as [`eval-vid_segmentation.ipynb`](src/eval-vid_segmentation.ipynb) and [`eval-img_segmentation.ipynb`](src/eval-img_segmentation.ipynb) notebooks.


### Checkpoints
See [`checkpoints`](checkpoints) folder for available checkpoints.


### Acknowledgements

This repository builds on [MaskFormer](https://github.com/facebookresearch/MaskFormer), [MotionGrouping](https://github.com/charigyang/motiongrouping), [unsupervised-image-segmentation](https://github.com/lukemelas/unsupervised-image-segmentation), and [dino-vit-features](https://github.com/ShirAmir/dino-vit-features).

### Citation   
```
@inproceedings{choudhury+karazija22gwm, 
    author    = {Choudhury, Subhabrata and Karazija, Laurynas and Laina, Iro and Vedaldi, Andrea and Rupprecht, Christian}, 
    booktitle = {British Machine Vision Conference (BMVC)}, 
    title     = {{G}uess {W}hat {M}oves: {U}nsupervised {V}ideo and {I}mage {S}egmentation by {A}nticipating {M}otion}, 
    year      = {2022}, 
}
```   