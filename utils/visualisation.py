import colorsys

import torch
import numpy as np
from cvbase.optflow.visualize import flow2rgb


def flow2rgb_torch(x):
    return torch.from_numpy(flow2rgb(x.permute(1, 2, 0).numpy())).permute(2, 0, 1)


def create_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=np.int64)
    colormap[0] = [0, 0, 0]
    colormap[1] = [166, 206, 227]
    colormap[2] = [31, 120, 180]
    colormap[3] = [178, 223, 138]
    colormap[4] = [51, 160, 44]
    colormap[5] = [251, 154, 153]
    colormap[6] = [227, 26, 28]
    colormap[7] = [253, 191, 111]
    colormap[8] = [255, 127, 0]
    colormap[9] = [202, 178, 214]
    colormap[10] = [106, 61, 154]
    colormap[11] = [255, 255, 153]
    colormap[12] = [177, 89, 40]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return torch.from_numpy(colormap).long()
