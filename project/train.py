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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

import video_consistent

from tqdm import tqdm

import todos
import pdb  # For debug

class Counter(object):
    """Class Counter."""
    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rgb_to_gray(image):
    gray_image = (0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] +
                  0.114 * image[:, 2, :, :])
    gray_image = gray_image.unsqueeze(1)

    return gray_image

def compute_gradient_loss(pred, gt):
    assert pred.shape == gt.shape, "a and b must have the same shape"

    pred = rgb_to_gray(pred)
    gt = rgb_to_gray(gt)

    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        dtype=pred.dtype, device=pred.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        dtype=pred.dtype, device=pred.device)

    gradient_a_x = F.conv2d(pred.repeat(1,3,1,1), 
        sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    gradient_a_y = F.conv2d(pred.repeat(1,3,1,1), 
        sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    # gradient_a_magnitude = torch.sqrt(gradient_a_x ** 2 + gradient_a_y ** 2)

    gradient_b_x = F.conv2d(gt.repeat(1,3,1,1), 
        sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    gradient_b_y = F.conv2d(gt.repeat(1,3,1,1), 
        sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3

    pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
    gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)

    # gradient_difference = torch.abs(pred_grad - gt_grad).mean(dim=1,keepdim=True)
    gradient_difference = torch.abs(pred_grad - gt_grad).mean()
    return gradient_difference

def train_epoch(loader, model, optimizer, device, tag="train"):
    """Trainning model ..."""

    total_loss = Counter()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    mse_loss = nn.MSELoss(reduction='mean')

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for batch in loader:
            # "rgbs", "grid", "tseq", "idxs"
            # Transform data to device
            rgbs = batch["rgbs"].to(device) # [2, 3, 720, 1280]
            grid = batch["grid"].to(device) # [2, 921600, 2]
            tseq = batch["tseq"].to(device) # [2, 1]

            count = len(rgbs) 
            # assert count == 1, "Current only support 1 batch"

            outputs = model(grid, tseq) # size() -- outputs.size() -- [2, 3, 921600]
            outputs = outputs.to(torch.float32)

            # Statics
            loss = mse_loss(rearrange(rgbs, 'b c h w -> b c (h w)'), outputs)
            pred_rgbs = outputs.reshape(rgbs.size()).to(torch.float32)
            grad_loss = compute_gradient_loss(pred_rgbs, rgbs).mean()
            loss = loss + 0.1 * grad_loss
            total_loss.update(loss.item(), 1) # 1 for mse_loss reduce mean

            # psnr = -10.0 * math.log10(loss.item() + 1e-5)
            psnr = -10.0 * math.log10(total_loss.avg + 1e-5)
            t.set_postfix(loss="{:.6f}, psnr={:.3f}".format(total_loss.avg, psnr))
            t.update(count)

            # Optimizer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return psnr

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, default="images", help="Input image directory")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="Checkpoint file")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--psnr", type=float, default=40.0, help="Stop training when we got PSNR")
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Step 1: get data loader
    train_ds = video_consistent.get_dataset(args.input)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)

    # Step 2: get net
    net, device = video_consistent.get_model(args.checkpoint)
    net.train()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Construct Optimizer and Learning Rate Scheduler
    # ***
    # ************************************************************************************/
    #
    lr_decay_step=[args.epochs*1//4, args.epochs*2//4, args.epochs*3//4]
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, eps=1e-8, weight_decay=0.0)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.5)

    last_psnr = 0.0
    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {:.8f} ...".format(epoch + 1, args.epochs, lr_scheduler.get_last_lr()[0]))
        last_psnr = train_epoch(train_dl, net, optimizer, device, tag="train")

        lr_scheduler.step()

        #
        # /************************************************************************************
        # ***
        # ***    MS: Define Save Model Strategy
        # ***
        # ************************************************************************************/
        #
        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1) or (last_psnr >= args.psnr):
            print(f"Saving model to {args.checkpoint} ...")
            model_dict = {}
            model_dict['psnr'] = last_psnr
            model_dict['frames'] = train_ds.frames
            model_dict['height'] = train_ds.height
            model_dict['width'] = train_ds.width
            model_dict['weight'] = net.state_dict()
            torch.save(model_dict, args.checkpoint)
            torch.save(net.xyt_grid.state_dict(), "/tmp/xyz.pth")

            net.update(train_ds.frames, train_ds.height, train_ds.width)

            # Create atlas image
            time = 0.0 # 0.0 - 1.0
            rgbs = video_consistent.image_sample(net, device, time)
            todos.data.save_tensor(rgbs, f"{args.output}/atlas.png")
            net.train()

        if (last_psnr >= args.psnr):
            break # Stop training for got expected quality

    todos.model.reset_device()

