"""Model trainning."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024, All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 20 Mar 2024 11:50:36 AM CST
# ***
# ************************************************************************************/
#

import os
import math
import argparse

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from video_consistent.dataset import make_image_grid
from video_consistent.network import VideoConsistenModel

from tqdm import tqdm

import todos
import pdb  # For debug


if __name__ == "__main__":
    """Trainning model."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("--canonical", type=str, default="", help="canonocal image file")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")

    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Step 1: get net
    assert os.path.exists(args.checkpoint), f"Model file {args.checkpoint} not exit."

    device = todos.model.get_device()
    net = VideoConsistenModel()
    net = net.to(device)
    print(f"Loading {args.checkpoint} ...")
    sd = torch.load(args.checkpoint)
    net.load_state_dict(sd['weight'])
    n_frames = sd['frames']
    frame_height = sd['height']
    frame_width = sd['width']
    print(f"model psnr={sd['psnr']:.3f}, frames: {n_frames}, {frame_height} x {frame_height}")
    net.eval()

    # Step 2:
    #
    # /************************************************************************************
    # ***
    # ***    MS: Construct Optimizer and Learning Rate Scheduler
    # ***
    # ************************************************************************************/
    #
    all_tseq = torch.linspace(0, 1, n_frames).unsqueeze(-1).to(device)
    grid = make_image_grid(frame_width, frame_height).unsqueeze(0).to(device)

    progress_bar = tqdm(total=n_frames)

    if len(args.canonical) > 0:
        c_image = cv2.cvtColor(cv2.imread(args.canonical), cv2.COLOR_BGR2RGB)
        c_image = torch.from_numpy(c_image.astype(np.float32)/255.0).unsqueeze(0)
        c_image = c_image.to(device)
    else:
        c_image = None

    for i in range(n_frames):
        progress_bar.update(1)

        output_filename = f"{args.output_dir}/{i:06d}.png"
        tseq = all_tseq[i].unsqueeze(0)
        # tensor [tseq] size: [1, 1], min: 0.0, max: 0.0, mean: 0.0
        # tensor [grid] size: [1, 921600, 2], min: 0.0, max: 0.999219, mean: 0.499457

        if c_image is None:
            with torch.no_grad():
                ret = net.forward(tseq, grid, True)
        else: # Sample from c_image
            grid_new = rearrange(grid, 'b n c -> (b n) c') # size() -- [518400, 2]
            with torch.no_grad():
                deformed_grid = net.deform_pts(tseq, grid_new, True)  # [batch * num_pixels, 2]

            c_height, c_width = c_image.shape[1:3]
            grid_new = deformed_grid.clone()
            grid_new[..., 1] = (2 * deformed_grid[..., 0] - 1) * frame_height / c_height
            grid_new[..., 0] = (2 * deformed_grid[..., 1] - 1) * frame_width / c_width
            # ------------------------------------------------------------
            ret = F.grid_sample(
                c_image.permute(0, 3, 1, 2),
                grid_new.unsqueeze(1).unsqueeze(0),
                mode='bilinear',
                padding_mode='border')
            # ------------------------------------------------------------
            # tensor [results] size: [518400, 3], min: 0.0, max: 0.996094, mean: 0.100389
            ret = ret.squeeze().permute(1,0)

        ret = ret.clamp(0.0, 1.0)
        # tensor [ret] size: [921600, 3], min: 0.0, max: 0.828125, mean: 0.106751

        ret = ret.float().cpu().numpy()
        image = rearrange(ret, '(h w) c -> h w c', h=frame_height, w=frame_width)
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_filename, image)
