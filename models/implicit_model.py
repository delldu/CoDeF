import torch
from torch import nn
import math
import tinycudann as tcnn
import todos
import pdb

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        # in_channels = 2
        # N_freqs = 8
        # self.N_freqs = N_freqs
        # self.in_channels = in_channels
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



class AnnealedHash(nn.Module):
    def __init__(self, in_channels, annealed_step, annealed_begin_step=0):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        # in_channels = 2
        # annealed_step = 4000
        # annealed_begin_step = 4000

        self.N_freqs = 16
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step
        index = torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.index = index.view(-1, 1).repeat(1, 2).view(-1)
        # ==> pdb.set_trace() for train

    def forward(self, x_embed, step):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        """
        if self.annealed_begin_step == 0:
            # calculate the w for each freq bands
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = self.N_freqs * (step - self.annealed_begin_step) / float(
                    self.annealed_step)

        w = (1.0 - torch.cos(math.pi * torch.clamp(alpha * torch.ones_like(self.index) - self.index, 0, 1))) / 2

        out = x_embed * w.to(x_embed.device)

        return out



class ImplicitVideo_Hash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=2,
                                     encoding_config=config["encoding"])
        # (Pdb) config["encoding"]
        # {'otype': 'HashGrid', 'n_levels': 16, 'n_features_per_level': 2, 'log2_hashmap_size': 19, 
        #     'base_resolution': 16, 'per_level_scale': 1.44}
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims + 2, # 34
                                    n_output_dims=3,
                                    network_config=config["network"])
        # (Pdb) config["network"]
        # {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 64, 
        #     'n_hidden_layers': 2}
        # ==> pdb.set_trace()

    def forward(self, x):
        # tensor [x] size: [819200, 2], min: 0.083333, max: 0.916016, mean: 0.499548

        input = x
        input = self.encoder(input)
        input = torch.cat([x, input], dim=-1)
        # weight = torch.ones(input.shape[-1], device=x.device)
        # # pdb.set_trace()
        # x = self.decoder(weight * input)
        x = self.decoder(input)

        # tensor [x] size: [819200, 3], min: -0.023834, max: 0.862793, mean: 0.112812
        return x


class Deform_Hash3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=3,
                                     encoding_config=config["encoding_deform3d"])
        # (Pdb) config["encoding_deform3d"]
        # {'otype': 'HashGrid', 'n_levels': 16, 'n_features_per_level': 2, 'log2_hashmap_size': 19, 
        #     'base_resolution': 16, 'per_level_scale': 1.38}

        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims + 3, # 35
                                    n_output_dims=2,
                                    network_config=config["network_deform"])
        # (Pdb) config["network_deform"]
        # {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 64, 
        #     'n_hidden_layers': 8}

        # (Pdb) self.encoder.forward.__code__ ---------------------
        # <file "/home/dell/miniconda3/envs/python39/lib/python3.9/site-packages/
        # tinycudann-1.7-py3.9-linux-x86_64.egg/tinycudann/modules.py", line 176>
        # (Pdb) self.decoder.forward.__code__ ---------------------
        # <file "/home/dell/miniconda3/envs/python39/lib/python3.9/site-packages/
        # tinycudann-1.7-py3.9-linux-x86_64.egg/tinycudann/modules.py", line 176>

    def forward(self, x, step=0, aneal_func=None):
        # tensor [x] size: [819200, 3], min: -0.166667, max: 1.165625, mean: 0.335629

        input = x
        input = self.encoder(input)
        if aneal_func is not None: # False
            pdb.set_trace() # for train, aneal_func ---- AnnealedHash()
            input = torch.cat([x, aneal_func(input,step)], dim=-1)
        else:
            input = torch.cat([x, input], dim=-1)

        # weight = torch.ones(input.shape[-1], device=x.device)
        # # pdb.set_trace()
        # x = self.decoder(weight * input) / 5

        x = self.decoder(input) / 5

        # todos.debug.output_var("x", x)
        # tensor [x] size: [819200, 2], min: -0.282471, max: 0.141968, mean: 0.038298
        return x


class Deform_Hash3d_Warp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Deform_Hash3d = Deform_Hash3d(config)
        # ==> pdb.set_trace()

    def forward(self, xyt_norm, step=0,aneal_func=None):
        x = self.Deform_Hash3d(xyt_norm,step=step, aneal_func=aneal_func)

        return x
