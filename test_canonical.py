import os
import json
import cv2
import numpy as np

from einops import rearrange
# from einops import repeat
from pathlib import Path
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader

from datasets import dataset_dict

from models.implicit_model import Embedding
from models.implicit_model import Deform_Hash3d_Warp

from utils import load_ckpt
from utils import VideoVisualizer

from opt import get_opts

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import todos
import pdb

class ImplicitVideoSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # hparams = Namespace(root_dir='all_sequences/white_smoke/white_smoke', 
        #     canonical_dir='all_sequences/white_smoke/base_control', mask_dir=None, flow_dir=None, 
        #     dataset_name='video', img_wh=[960, 540], canonical_wh=[1280, 640], 
        #     ref_idx=None, encode_w=True, batch_size=1, num_steps=10000, valid_iters=30, 
        #     valid_batches=0, save_model_iters=2000, gpus=[0], test=True, ckpt_path=None, 
        #     prefixes_to_ignore=['loss'], weight_path='all_sequences/white_smoke/base/white_smoke.ckpt', 
        #     model_save_path='ckpts', log_save_path='logs/test_all_sequences/white_smoke', exp_name='base', 
        #     optimizer='adam', lr=0.001, momentum=0.9, weight_decay=0, lr_scheduler='steplr', 
        #     decay_step=[2500, 5000, 7500], decay_gamma=0.5, warmup_multiplier=1.0, warmup_epochs=0, 
        #     annealed=False, annealed_begin_step=4000, annealed_step=4000, flow_loss=0, bg_loss=0.003, 
        #     grad_loss=0.1, flow_step=-1, ref_step=-1, self_bg=True, sigmoid_offset=0, 
        #     save_deform=False, save_video=True, fps=30, deform_D=6, deform_W=128, vid_D=8, vid_W=256, 
        #     N_vocab_w=200, N_w=8, N_xyz_w=[8], vid_hash=True, deform_hash=True, 
        #     config='configs/white_smoke/base.yaml')

        self.save_hyperparameters(hparams)
        if hparams.save_video: # True
            self.video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.raw_video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.dual_video_visualizer = VideoVisualizer(fps=hparams.fps)

        # self.embeddings = {'xyz': Embedding(2, 8)}
        # (Pdb) self.embeddings
        # {'xyz': Embedding(), 'xyz_w': [Embedding()], 'w_0': Embedding(200, 8)}

        self.models = {}

        # Construct normalized meshgrid.
        h = self.hparams.img_wh[1]
        w = self.hparams.img_wh[0]

        self.num_models = 1

        # Decide the number of deformable mlp.
        if hparams.encode_w: # True
            # Multiple deformation MLP.
            # Progressive Training for the Deformation (Annealed PE).
            # No trainable parameters.
            # self.embeddings['xyz_w'] = []
            # assert (isinstance(self.hparams.N_xyz_w, list))
            # in_channels_xyz = []
            # for i in range(self.num_models):
            #     N_xyz_w = self.hparams.N_xyz_w[i]
            #     in_channels_xyz += [2 + 2 * N_xyz_w * 2]
            #     # self.embeddings['xyz_w'] += [Embedding(2, N_xyz_w)]

            for i in range(self.num_models):
                # embedding_w = torch.nn.Embedding(hparams.N_vocab_w, hparams.N_w) # Embedding(200, 8)
                # torch.nn.init.uniform_(embedding_w.weight, -0.05, 0.05)
                # load_ckpt(embedding_w, hparams.weight_path, model_name=f'w_{i}')
                # # self.embeddings[f'w_{i}'] = embedding_w

                # Add warping field mlp.
                with open('configs/hash.json') as f:
                    config = json.load(f)
                warping_field = Deform_Hash3d_Warp(config=config)

                load_ckpt(warping_field, # warping_field -- Deform_Hash3d_Warp(...)
                          hparams.weight_path, # 'all_sequences/white_smoke/base/white_smoke.ckpt'
                          model_name=f'warping_field_{i}')
                self.models[f'warping_field_{i}'] = warping_field

        # for key in self.embeddings:
        #     setattr(self, key, self.embeddings[key])
        for key in self.models:
            setattr(self, key, self.models[key])

        # (Pdb) self.models.keys() -- ['warping_field_0']
        # self.models['warping_field_0'] -- Deform_Hash3d_Warp(...)

    def deform_pts(self, ts_w, grid, i=0):
        # ts_w = tensor([[1]], device='cuda:0')
        # grid = tensor([[0.000000, 0.000000],
        #         [0.000000, 0.001042],
        #         [0.000000, 0.002083],
        #         ...,
        #         [0.998148, 0.996875],
        #         [0.998148, 0.997917],
        #         [0.998148, 0.998958]], device='cuda:0')
        # encode_w = True
        # step = 0
        # i = 0 -- model no
        ts_w_norm = ts_w / self.seq_len
        ts_w_norm = ts_w_norm.repeat(grid.shape[0], 1)
        input_xyt = torch.cat([grid, ts_w_norm], dim=-1)
        # ----------------------------------------------------------------------------
        # self.models.keys() -- ['warping_field_0']
        deform = self.models[f'warping_field_{i}'](input_xyt) # size() -- [518400, 3]
        deformed_grid = deform + grid

        # tensor [deformed_grid] size: [518400, 2], min: 0.024697, max: 1.060139, mean: 0.538712
        return deformed_grid

    def forward(self,
                ts_w,
                grid, # [1, 518400, 2]
                flows=None):


        # grid -> positional encoding
        # ts_w -> embedding
        # ---------------------------------------------------
        # ts_w = tensor([[1]], device='cuda:0')
        # grid = tensor([[[0.000000, 0.000000],
        #          [0.000000, 0.001042],
        #          [0.000000, 0.002083],
        #          ...,
        #          [0.998148, 0.996875],
        #          [0.998148, 0.997917],
        #          [0.998148, 0.998958]]], device='cuda:0')
        # encode_w = True
        # step = 0
        # flows = None

        grid = rearrange(grid, 'b n c -> (b n) c') # size() -- [518400, 2]
        results_list = []
        for i in range(self.num_models): # self.num_models === 1
            deformed_grid = self.deform_pts(ts_w, grid, i)  # [batch * num_pixels, 2]
            # deform_list.append(deformed_grid)
            pe_deformed_grid = (deformed_grid + 0.3) / 1.6

            w, h = self.img_wh
            canonical_img = self.canonical_img.squeeze(0)
            h_c, w_c = canonical_img.shape[1:3]
            grid_new = deformed_grid.clone()
            grid_new[..., 1] = (2 * deformed_grid[..., 0] - 1) * h / h_c
            grid_new[..., 0] = (2 * deformed_grid[..., 1] - 1) * w / w_c
            # ------------------------------------------------------------
            results = torch.nn.functional.grid_sample(
                canonical_img[i:i + 1].permute(0, 3, 1, 2),
                grid_new.unsqueeze(1).unsqueeze(0),
                mode='bilinear',
                padding_mode='border')
            # ------------------------------------------------------------
            # tensor [results] size: [518400, 3], min: 0.0, max: 0.996094, mean: 0.100389
            results = results.squeeze().permute(1,0)

            results_list.append(results)

        ret = edict(rgbs=results_list)

        return ret

    def setup(self, stage):
        pass

    def test_dataloader(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'root_dir': self.hparams.root_dir,
            'img_wh': tuple(self.hparams.img_wh), # (960, 540)
            'mask_dir': self.hparams.mask_dir,
            'canonical_wh': self.hparams.canonical_wh, # [1280, 640]
            'canonical_dir': self.hparams.canonical_dir, # 'all_sequences/white_smoke/base_control'
            'test': self.hparams.test # True
        }
        self.train_dataset = dataset(split='train', **kwargs)

        return DataLoader(
            self.train_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time.
            pin_memory=True)

    def test_step(self, batch, batch_idx):
        # ---------------------------------------
        # batch is dict:
        #     tensor [rgbs] size: [1, 540, 960, 3], min: 0.0, max: 0.871094, mean: 0.105189
        #     tensor [canonical_img] size: [1, 1, 640, 1280, 3], min: 0.0, max: 0.996094, mean: 0.1106
        #     tensor [ts_w] size: [1, 1], min: 1.0, max: 1.0, mean: 1.0
        #     tensor [grid] size: [1, 518400, 2], min: 0.0, max: 0.998958, mean: 0.499277
        #     [canonical_wh] type: <class 'list'>
        #     [img_wh] type: <class 'list'>
        #     [masks] type: <class 'list'>
        #     tensor [flows] size: [1], min: -100000.0, max: -100000.0, mean: -100000.0
        #     tensor [grid_c] size: [1, 819200, 2], min: -0.166667, max: 1.165625, mean: 0.499277
        #     tensor [reference] size: [1], min: -100000.0, max: -100000.0, mean: -100000.0
        #     tensor [seq_len] size: [1], min: 120.0, max: 120.0, mean: 120.0
        # batch['masks'] is list: len = 1
        #     tensor [item] size: [1, 540, 960, 1], min: 1.0, max: 1.0, mean: 1.0
        # batch['img_wh'] is list: len = 2
        #     tensor [item] size: [1], min: 960.0, max: 960.0, mean: 960.0
        #     tensor [item] size: [1], min: 540.0, max: 540.0, mean: 540.0
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        W, H = self.hparams.img_wh
        self.seq_len = batch['seq_len']
        if self.hparams.canonical_dir is not None: # True
            self.canonical_img = batch['canonical_img']
            self.img_wh = batch['img_wh']

        save_dir = os.path.join('results',
                                self.hparams.root_dir.split('/')[0],
                                self.hparams.root_dir.split('/')[1],
                                self.hparams.exp_name)
        sample_name = self.hparams.root_dir.split('/')[1]
        test_dir = f'{save_dir}_transformed'
        video_name = f'{sample_name}_{self.hparams.exp_name}_transformed'
        Path(test_dir).mkdir(parents=True, exist_ok=True)

        if batch_idx > 0 and self.hparams.save_video: # self.hparams.save_video -- True
            self.video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}.mp4'))
            self.raw_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_raw.mp4'))
            self.dual_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_dual.mp4'))

        # --------------- forward ------------------------
        ret_n = self.forward(ts_w, grid)
        # ret_n['rgbs'] is list: len = 1
        #     tensor [item] size: [518400, 3], min: 0.0, max: 0.996094, mean: 0.100389

        img = np.zeros((H * W, 3), dtype=np.float32) # (518400, 3), (H, W) === (540, 960)
        for i in range(self.num_models):
            mk_n = rearrange(mk[i], 'b h w c -> (b h w) c')
            mk_n = mk_n.sum(dim=-1) > 0.05
            mk_n = mk_n.cpu().numpy() # shape -- (518400,)

            results = ret_n.rgbs[i]
            results = results.cpu().numpy()  # (3, H, W)
            img[mk_n] = results[mk_n]

        img = rearrange(img, '(h w) c -> h w c', h=H, w=W)
        img = img * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{test_dir}/{batch_idx:05d}.png', img)

        if batch_idx > 0 and self.hparams.save_video: # self.hparams.save_video === True
            img = img[..., ::-1]
            self.video_visualizer.add(img)
            rgbs = batch['rgbs'].view(H, W, 3).cpu().numpy() * 255
            rgbs = rgbs.astype(np.uint8)
            self.raw_video_visualizer.add(rgbs)
            dual_img = np.concatenate((rgbs, img), axis=1)
            self.dual_video_visualizer.add(dual_img)


    def on_test_epoch_end(self):
        if self.hparams.save_video:
            # ==> pdb.set_trace()
            self.video_visualizer.save()
            self.raw_video_visualizer.save()
            self.dual_video_visualizer.save()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

def main(hparams):
    system = ImplicitVideoSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{hparams.model_save_path}/{hparams.exp_name}',
        filename='{step:d}',
        mode='max',
        save_top_k=-1,
        every_n_train_steps=hparams.save_model_iters,
        save_last=True)

    logger = TensorBoardLogger(save_dir=hparams.log_save_path,
                               name=hparams.exp_name)

    trainer = Trainer(max_steps=hparams.num_steps,
                      precision=16 if hparams.vid_hash == True else 32,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      accelerator='gpu',
                      devices=hparams.gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if len(hparams.gpus) == 1 else None,
                      val_check_interval=hparams.valid_iters,
                      limit_val_batches=hparams.valid_batches,
                      strategy="ddp") # "ddp_find_unused_parameters_true"

    trainer.test(system, dataloaders=system.test_dataloader())


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
