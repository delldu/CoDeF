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

# class Embedding(nn.Module):
#     def __init__(self, in_channels=2, N_freqs=8):
#         """
#         Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
#         in_channels: number of input channels (3 for both xyz and direction)
#         """
#         super().__init__()
#         self.funcs = [torch.sin, torch.cos] ###
#         self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
#         # [  1.,   2.,   4.,   8.,  16.,  32.,  64., 128.]

#     def forward(self, x):
#         """
#         Embeds x to (x, sin(2^k x), cos(2^k x), ...)
#         """
#         out = [x]
#         for freq in self.freq_bands:
#             for func in self.funcs:
#                 out += [func(freq*x)]

#         return torch.cat(out, -1)


class VideoHash(nn.Module):
    '''
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.44
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    },
    '''
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
        input = self.encoder(input) # [921600, 32]
        input = torch.cat([x, input], dim=1) # [921600, 34]
        x = self.decoder(input)

        # tensor [x] size: [819200, 3], min: -0.023834, max: 0.862793, mean: 0.112812
        return x


class PlaneHash(nn.Module):
    '''
    "encoding_deform3d": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.38
    },
    "network_deform": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 8
    }
    '''
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

    def forward(self, x):
        # tensor [x] size: [819200, 3], min: -0.166667, max: 1.165625, mean: 0.335629

        input = x
        input = self.encoder(input) # [921600, 32]
        input = torch.cat([x, input], dim=1) # size() -- [921600, 35]

        x = self.decoder(input) / 5.0
        # tensor [x] size: [819200, 2], min: -0.282471, max: 0.141968, mean: 0.038298
        return x


class VideoConsistenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.plane_hash = PlaneHash() # xyt->grid
        self.video_hash = VideoHash() # grid->rgb

        # Place holder
        self.frames = 1
        self.height = 1
        self.wdith = 1

    def update(self, n, h, w):
        self.frames = n
        self.height = h
        self.width = w

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            print(f"Loading {file_path} ...")
            sd = torch.load(file_path)
            self.load_state_dict(sd['weight'])
            self.update(sd['frames'], sd['height'], sd['width'])
            print(f"model psnr={sd['psnr']:.3f}, frames: {self.frames}, {self.height} x {self.width}")

    def deform_xyt(self, one_grid, one_time, enable_warp: bool = True):
        # one_grid - [921600, 2]
        # one_time.size() -- [1]
        if enable_warp:
            HW, C = one_grid.size()
            one_time = one_time.squeeze(0).repeat(HW, 1)
            input_xyt = torch.cat([one_grid, one_time], dim=1)
            deform = self.plane_hash(input_xyt) # size() -- [518400, 3]
            deformed_grid = deform + one_grid
        else:
            deformed_grid = one_grid

        # tensor [deformed_grid] size: [518400, 2], min: 0.024697, max: 1.060139, mean: 0.538712
        return deformed_grid # size() -- [H*W, 2]

    def forward(self, grids, times, enable_warp: bool = True):
        # grids - [4, 921600, 2]
        # grids = tensor([[[0.000000, 0.000000],
        #          [0.000000, 0.001042],
        #          [0.000000, 0.002083],
        #          ...,
        #          [0.998148, 0.996875],
        #          [0.998148, 0.997917],
        #          [0.998148, 0.998958]]], device='cuda:0')
        # times.size() -- [4, 1]

        B, HW, C = grids.size()
        # tiny cudann ONLY support one batch, overcome with loop !!!
        rgb_predict = torch.zeros(B, HW, C + 1).to(grids.device)
        for i in range(B):
            one_grid = grids[i]
            one_time = times[i]
            one_deform = self.deform_xyt(one_grid, one_time, enable_warp)
            pe_one_deform = (one_deform + 0.3) / 1.6 # [B, H*W, 2]
            rgb_predict[i] = self.video_hash(pe_one_deform) # [B, H*W, 3]

        return rgb_predict.permute(0, 2, 1) # [B, H*W, C] ==> [B, C, H*W]
