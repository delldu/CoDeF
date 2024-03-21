import torch
from einops import rearrange
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import pdb

def make_grid(h:int, w:int):
    grid = np.indices((h, w)).astype(np.float32)
    # normalize
    grid[0,:,:] = grid[0,:,:] / h
    grid[1,:,:] = grid[1,:,:] / w
    grid = torch.from_numpy(rearrange(grid, 'c h w -> (h w) c'))
    return grid # size() -- [h * w, 2]

def load_image(image_path: str, w: int, h: int):
    input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Need resize ?
    if input_image.shape[0] != h or input_image.shape[1] != w:
        input_image = cv2.resize(input_image, (w, h), interpolation=cv2.INTER_AREA)

    input_image = torch.from_numpy(np.array(input_image).astype(np.float32) / 255.0)
    return input_image.permute(2, 0, 1) # hxwxc --> cxhxw


class VideoDataset(Dataset):
    def __init__(self, root_dir):
        all_images_path = sorted(glob.glob(f"{root_dir}/*.png"))
        assert len(all_images_path) > 0, f"{root_dir} has any png image file"
        first_image = cv2.cvtColor(cv2.imread(all_images_path[0]), cv2.COLOR_BGR2RGB)
        h = first_image.shape[0]
        w = first_image.shape[1]
        self.all_images = [load_image(ip, w, h) for ip in all_images_path] # [C, H, W]
        self.grid = make_grid(h, w) # [H * W, 2]
        # Normal time sequence
        self.tseq = torch.linspace(0, 1, len(self.all_images)).unsqueeze(-1)

        # Save frame heigh and width
        self.height = h
        self.width = w
        self.frames = len(self.all_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        return {
            "rgbs": self.all_images[idx],
            "grid": self.grid,
            "tseq": self.tseq[idx],
            "idxs": idx,
        }
