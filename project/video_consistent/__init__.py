"""Video Consistent Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import torch
import torch.nn.functional as F

from tqdm import tqdm

from . import network
from . import dataset

import todos

import pdb

def get_dataset(input_dir):
    """Get dataset."""
    return dataset.VideoDataset(input_dir)

def get_model(checkpoint):
    """Create model."""
    device = todos.model.get_device()
    net = network.VideoConsisten()
    net.load_weights(checkpoint)
    net = net.to(device)

    print(f"Running model on {device} ...")

    return net, device


def video_restruct(net, device, output_dir):
    # Create directory to store result
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_frames = net.frames
    all_tseq = torch.linspace(0, 1, net.frames).unsqueeze(1).to(device)
    grid = dataset.make_grid(net.height, net.width).unsqueeze(0).to(device)

    progress_bar = tqdm(total=n_frames)

    net.eval()
    for i in range(n_frames):
        progress_bar.update(1)

        output_filename = f"{output_dir}/{i:06d}.png"
        tseq = all_tseq[i].unsqueeze(0)
        # tensor [tseq] size: [1, 1], min: 0.0, max: 0.0, mean: 0.0
        # tensor [grid] size: [1, 921600, 2], min: 0.0, max: 0.999219, mean: 0.499457

        with torch.no_grad():
            ret = net.forward(grid, tseq)
        # tensor [ret] size: [1, 3, 921600], min: 0.0, max: 0.828125, mean: 0.106751
        rgbs = ret.reshape(1, 3, net.height, net.width)
        rgbs = rgbs.clamp(0.0, 1.0)
        todos.data.save_tensor(rgbs, output_filename)

    todos.model.reset_device()

def video_sample(ref_filename, net, device, output_dir):
    c_image = todos.data.load_tensor(ref_filename).to(device)
    B, C, H, W = c_image.size()

    # Create directory to store result
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_frames = net.frames
    all_tseq = torch.linspace(0, 1, net.frames).unsqueeze(1).to(device)
    grid = dataset.make_grid(net.height, net.width).unsqueeze(0).to(device)

    progress_bar = tqdm(total=n_frames)

    net.eval()
    for i in range(n_frames):
        progress_bar.update(1)

        output_filename = f"{output_dir}/{i:06d}.png"
        tseq = all_tseq[i].unsqueeze(0)
        # tensor [tseq] size: [1, 1], min: 0.0, max: 0.0, mean: 0.0
        # tensor [grid] size: [1, 921600, 2], min: 0.0, max: 0.999219, mean: 0.499457
        one_grid = grid.squeeze(0)
        one_time = all_tseq[i]
        with torch.no_grad():
            deformed_grid = net.deform_xyt(one_grid, one_time)  # [batch * num_pixels, 2]

        grid_new = deformed_grid.clone()
        grid_new[..., 1] = (2 * deformed_grid[..., 0] - 1) * net.height / H
        grid_new[..., 0] = (2 * deformed_grid[..., 1] - 1) * net.width / W
        grid_new = grid_new.reshape(1, net.height, net.width, 2)

        # ------------------------------------------------------------
        rgbs = F.grid_sample(c_image, grid_new, mode='bilinear', padding_mode='border')
        # ------------------------------------------------------------
        # tensor [rgbs] size: [1, 3, H, W], min: 0.0, max: 0.996094, mean: 0.100389

        rgbs = rgbs.clamp(0.0, 1.0)
        # tensor [rgbs] size: [921600, 3], min: 0.0, max: 0.828125, mean: 0.106751
        todos.data.save_tensor(rgbs, output_filename)

    todos.model.reset_device()