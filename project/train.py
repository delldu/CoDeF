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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

from video_consistent.dataset import VideoDataset, make_image_grid
from video_consistent.network import VideoConsistenModel

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

    gradient_difference = torch.abs(pred_grad - gt_grad).mean(dim=1,keepdim=True)

    return gradient_difference

def train_epoch(loader, model, optimizer, device, tag="train"):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = nn.MSELoss(reduction='mean')

    correct = 0
    total = 0

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for batch in loader:
            # "rgbs", "grid", "tseq", "idxs"
            # Transform data to device
            rgbs = batch["rgbs"].to(device)
            grid = batch["grid"].to(device)
            tseq = batch["tseq"].to(device)

            count = len(rgbs)
            assert count == 1, "Current only support 1 batch"

            # Transform data to device
            outputs = model(tseq, grid) # size() -- outputs.size() -- 921600, 3
            rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')

            # Statics
            loss = loss_function(rgbs_flattend, outputs.to(torch.float32))
            psnr = -20.0 * math.log10(math.sqrt(total_loss.avg + 1e-5))


            total_loss.update(loss.item(), count)
            t.set_postfix(loss="{:.6f}, psnr={:.3f}".format(total_loss.avg, psnr))
            t.update(count)

            # Optimizer
            img_pred = outputs.reshape(rgbs.size()).to(torch.float32)
            grad_loss = compute_gradient_loss(rgbs.permute(0, 3, 1, 2),
                                              img_pred.permute(0, 3, 1, 2)).mean()
            loss = loss + 0.1 * grad_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return psnr

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, default="images", help="input image directory")

    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Step 1: get data loader
    train_ds = VideoDataset(args.input_dir)
    # Now only support batch_size === 1
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

    # Step 2: get net
    device = todos.model.get_device()
    net = VideoConsistenModel()
    net = net.to(device)
    if os.path.exists(args.checkpoint):
        print(f"Loading {args.checkpoint} ...")
        sd = torch.load(args.checkpoint)
        net.load_state_dict(sd['weight'])
        print(f"model psnr={sd['psnr']:.3f}, frames: {sd['frames']}, {sd['height']} x {sd['width']}")

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
        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            print(f"Saving model to {args.checkpoint} ...")
            model_dict = {}
            model_dict['psnr'] = last_psnr
            model_dict['frames'] = train_ds.frames
            model_dict['height'] = train_ds.height
            model_dict['width'] = train_ds.width
            model_dict['weight'] = net.state_dict()
            torch.save(model_dict, args.checkpoint)

    # Create canonical image
    c_h = train_ds.height + 64
    c_w = train_ds.width + 64
    canonical_filename = f"{args.output_dir}/canonical.png"
    grid = make_image_grid(c_w, c_h).unsqueeze(0).to(device)
    tseq = torch.zeros((1)).unsqueeze(0).to(device)
    net.eval()

    with torch.no_grad():
        ret = net.forward(tseq, grid, False)

    ret = ret.clamp(0.0, 1.0)
    # tensor [ret] size: [921600, 3], min: 0.0, max: 0.828125, mean: 0.106751

    ret = ret.float().cpu().numpy()
    image = rearrange(ret, '(h w) c -> h w c', h=c_h, w=c_w)
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(canonical_filename, image)
