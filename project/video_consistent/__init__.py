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
    net = network.VideoConsistenModel()
    net.load_weights(checkpoint)
    net = net.to(device)

    print(f"Running model on {device} ...")

    return net, device


def create_canonical_image(net, device, canonical_filename):
    net.eval()

    c_h = net.height
    c_w = net.width
    grid = dataset.make_grid(c_h, c_w).unsqueeze(0).to(device) # [1, H*W, 2]
    tseq = torch.zeros((1)).unsqueeze(0).to(device) # [1, 1]

    with torch.no_grad():
        ret = net.forward(grid, tseq, False) # [B, C, H*W]

    ret = ret.reshape(1, 3, c_h, c_w).clamp(0.0, 1.0)
    # tensor [ret] size: [1, 3, c_h, c_w], min: 0.0, max: 0.828125, mean: 0.106751
    todos.data.save_tensor(ret, canonical_filename)
    todos.model.reset_device()


def video_restruct(net, device, output_dir):
    # Create directory to store result
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net.eval()

    n_frames = net.frames
    all_tseq = torch.linspace(0, 1, net.frames).unsqueeze(1).to(device)
    grid = dataset.make_grid(net.height, net.width).unsqueeze(0).to(device)

    progress_bar = tqdm(total=n_frames)

    for i in range(n_frames):
        progress_bar.update(1)

        output_filename = f"{output_dir}/{i:06d}.png"
        tseq = all_tseq[i].unsqueeze(0)
        # tensor [tseq] size: [1, 1], min: 0.0, max: 0.0, mean: 0.0
        # tensor [grid] size: [1, 921600, 2], min: 0.0, max: 0.999219, mean: 0.499457

        with torch.no_grad():
            ret = net.forward(grid, tseq, True)
        ret = ret.reshape(1, 3, net.height, net.width)
        ret = ret.clamp(0.0, 1.0)
        # tensor [ret] size: [921600, 3], min: 0.0, max: 0.828125, mean: 0.106751
        todos.data.save_tensor(ret, output_filename)

    todos.model.reset_device()

def video_restruct_with_canonical_image(canonical_file, net, device, output_dir):
    c_image = todos.data.load_tensor(canonical_file).to(device)

    # Create directory to store result
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net.eval()

    n_frames = net.frames
    all_tseq = torch.linspace(0, 1, net.frames).unsqueeze(1).to(device)
    grid = dataset.make_grid(net.height, net.width).unsqueeze(0).to(device)

    progress_bar = tqdm(total=n_frames)

    for i in range(n_frames):
        progress_bar.update(1)

        output_filename = f"{output_dir}/{i:06d}.png"
        tseq = all_tseq[i].unsqueeze(0)
        # tensor [tseq] size: [1, 1], min: 0.0, max: 0.0, mean: 0.0
        # tensor [grid] size: [1, 921600, 2], min: 0.0, max: 0.999219, mean: 0.499457
        one_grid = grid.squeeze(0)
        one_time = all_tseq[i]
        with torch.no_grad():
            deformed_grid = net.deform_pts(one_grid, one_time, True)  # [batch * num_pixels, 2]

        B, C, H, W = c_image.size()
        grid_new = deformed_grid.clone()
        grid_new[..., 1] = (2 * deformed_grid[..., 0] - 1) * net.height / H
        grid_new[..., 0] = (2 * deformed_grid[..., 1] - 1) * net.width / W
        grid_new = grid_new.reshape(1, net.height, net.width, 2)

        # ------------------------------------------------------------
        ret = F.grid_sample(c_image, grid_new, mode='bilinear', padding_mode='border')
        # ------------------------------------------------------------
        # tensor [results] size: [1, 3, H, W], min: 0.0, max: 0.996094, mean: 0.100389
        # ret = ret.squeeze().permute(1,0)

        ret = ret.clamp(0.0, 1.0)
        # tensor [ret] size: [921600, 3], min: 0.0, max: 0.828125, mean: 0.106751
        todos.data.save_tensor(ret, output_filename)

    todos.model.reset_device()