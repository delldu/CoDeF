"""Model Define."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022-2024, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 22日 星期四 03:54:39 CST
# ***
# ************************************************************************************/
#
import os
import json

import torch
from torch import nn
import tinycudann as tcnn
from einops import rearrange

import todos
import pdb

class Embedding(nn.Module):
    def __init__(self, in_channels=2, N_freqs=8):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos] ###
        self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        # [  1.,   2.,   4.,   8.,  16.,  32.,  64., 128.]

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class VideoHash(nn.Module):
    def __init__(self):
        super().__init__()
        cdir = os.path.dirname(__file__)
        with open(f"{cdir}/hash.json") as f:
            config = json.load(f)

        self.encoder = tcnn.Encoding(
            n_input_dims=2, 
            encoding_config=config["encoding"],
        )
        self.decoder = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims + 2, # 34
            n_output_dims=3,
            network_config=config["network"],
        )


    def forward(self, x):
        # tensor [x] size: [819200, 2], min: 0.083333, max: 0.916016, mean: 0.499548

        input = x
        input = self.encoder(input)
        input = torch.cat([x, input], dim=1)
        x = self.decoder(input)

        # tensor [x] size: [819200, 3], min: -0.023834, max: 0.862793, mean: 0.112812
        return x


class DeformHash(nn.Module):
    def __init__(self):
        super().__init__()
        cdir = os.path.dirname(__file__)
        with open(f"{cdir}/hash.json") as f:
            config = json.load(f)

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=config["encoding_deform3d"],
        )
        self.decoder = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims + 3, # 35
            n_output_dims=2,
            network_config=config["network_deform"],
        )

    def forward(self, x, step=0):
        # tensor [x] size: [819200, 3], min: -0.166667, max: 1.165625, mean: 0.335629

        input = x
        input = self.encoder(input)
        input = torch.cat([x, input], dim=-1)

        x = self.decoder(input) / 5.0
        # tensor [x] size: [819200, 2], min: -0.282471, max: 0.141968, mean: 0.038298
        return x


class VideoConsistenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deform_hash = DeformHash() # rgb->grid
        self.video_hash = VideoHash() # grid->rgb

    def deform_pts(self, tseq, grid, enable_warp: bool = True):
        # grid - [921600, 2]
        # tseq.size() -- [1, 1]
        if enable_warp:
            tseq = tseq.repeat(grid.shape[0], 1)
            input_xyt = torch.cat([grid, tseq], dim=-1)
            deform = self.deform_hash(input_xyt) # size() -- [518400, 3]
            deformed_grid = deform + grid
        else:
            deformed_grid = grid

        # tensor [deformed_grid] size: [518400, 2], min: 0.024697, max: 1.060139, mean: 0.538712
        return deformed_grid

    def forward(self, tseq, grid, enable_warp: bool = True):
        # tseq.size() -- [4, 1]
        # tseq = tensor([[1]], device='cuda:0')

        # grid - [4, 921600, 2]
        # grid = tensor([[[0.000000, 0.000000],
        #          [0.000000, 0.001042],
        #          [0.000000, 0.002083],
        #          ...,
        #          [0.998148, 0.996875],
        #          [0.998148, 0.997917],
        #          [0.998148, 0.998958]]], device='cuda:0')

        grid = rearrange(grid, 'b n c -> (b n) c') # size() -- [921600, 2]
        deformed_grid = self.deform_pts(tseq, grid, enable_warp)  # [batch * num_pixels, 2]
        pe_deformed_grid = (deformed_grid + 0.3) / 1.6
        return self.video_hash(pe_deformed_grid)

