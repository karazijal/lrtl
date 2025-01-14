import torch


def get_meshgrid(resolution, device):
    grid_x, grid_y = torch.meshgrid(torch.arange(resolution[0]).float() / resolution[0],
                                    torch.arange(resolution[1]).float() / resolution[1], indexing='ij')
    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)
    return grid_x, grid_y
