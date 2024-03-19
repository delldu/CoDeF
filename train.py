import os
import json
import cv2
import numpy as np

from einops import rearrange
from einops import repeat
from pathlib import Path
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import dataset_dict
from losses import loss_dict
from losses import compute_gradient_loss

from models.implicit_model import ImplicitVideo_Hash
from models.implicit_model import Embedding
from models.implicit_model import AnnealedHash
from models.implicit_model import Deform_Hash3d_Warp

from utils import get_optimizer
from utils import get_scheduler
from utils import get_learning_rate
from utils import load_ckpt
from utils import VideoVisualizer

from opt import get_opts
from metrics import psnr

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
        self.color_loss = loss_dict['mse'](coef=1)
        if hparams.save_video: # True
            self.video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.raw_video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.dual_video_visualizer = VideoVisualizer(fps=hparams.fps)

        self.models_to_train=[]
        self.embeddings = {'xyz': Embedding(2, 8)}
        # (Pdb) self.embeddings
        # {'xyz': Embedding(), 'xyz_w': [Embedding()], 'w_0': Embedding(200, 8)}

        self.models = {}

        # Construct normalized meshgrid.
        h = self.hparams.img_wh[1]
        w = self.hparams.img_wh[0]
        self.h = h
        self.w = w

        if self.hparams.mask_dir: # False
            self.num_models = len(self.hparams.mask_dir)
        else:
            self.num_models = 1

        # Decide the number of deformable mlp.
        if hparams.encode_w: # True
            # Multiple deformation MLP.
            # Progressive Training for the Deformation (Annealed PE).
            # No trainable parameters.
            self.embeddings['xyz_w'] = []
            assert (isinstance(self.hparams.N_xyz_w, list))
            in_channels_xyz = []
            for i in range(self.num_models):
                N_xyz_w = self.hparams.N_xyz_w[i]
                in_channels_xyz += [2 + 2 * N_xyz_w * 2]
                self.embedding_hash = AnnealedHash(
                in_channels=2,
                annealed_step=hparams.annealed_step,
                annealed_begin_step=hparams.annealed_begin_step)
                self.embeddings['aneal_hash'] = self.embedding_hash

            for i in range(self.num_models): # self.num_models --- 1
                embedding_w = torch.nn.Embedding(hparams.N_vocab_w, hparams.N_w) # Embedding(200, 8)
                torch.nn.init.uniform_(embedding_w.weight, -0.05, 0.05)
                load_ckpt(embedding_w, hparams.weight_path, model_name=f'w_{i}')
                self.embeddings[f'w_{i}'] = embedding_w
                self.models_to_train += [self.embeddings[f'w_{i}']]

                # Add warping field mlp.
                with open('configs/hash.json') as f:
                    config = json.load(f)
                warping_field = Deform_Hash3d_Warp(config=config)

                load_ckpt(warping_field, # warping_field -- Deform_Hash3d_Warp(...)
                          hparams.weight_path, # 'all_sequences/white_smoke/base/white_smoke.ckpt'
                          model_name=f'warping_field_{i}')
                self.models[f'warping_field_{i}'] = warping_field

        # Set up the canonical model.
        if hparams.canonical_dir is None: # False
            for i in range(self.num_models):
                with open('configs/hash.json') as f:
                    config = json.load(f)
                implicit_video = ImplicitVideo_Hash(config=config)
                load_ckpt(implicit_video, hparams.weight_path,
                          f'implicit_video_{i}')
                self.models[f'implicit_video_{i}'] = implicit_video

        for key in self.embeddings:
            setattr(self, key, self.embeddings[key])
        for key in self.models:
            setattr(self, key, self.models[key])

        self.models_to_train += [self.models]
        # (Pdb) self.models.keys() --dict_keys(['warping_field_0'])
        # self.models['warping_field_0'] -- Deform_Hash3d_Warp(...)

    def deform_pts(self, ts_w, grid, encode_w, step=0, i=0):
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

        if hparams.deform_hash: # True
            ts_w_norm = ts_w / self.seq_len
            ts_w_norm = ts_w_norm.repeat(grid.shape[0], 1)
            input_xyt = torch.cat([grid, ts_w_norm], dim=-1)
            if 'aneal_hash' in self.embeddings.keys(): # False
                deform = self.models[f'warping_field_{i}'](
                    input_xyt,
                    step=step,
                    aneal_func=self.embeddings['aneal_hash'])
            else:
                deform = self.models[f'warping_field_{i}'](input_xyt) # size() -- [518400, 3]
            if encode_w:
                deformed_grid = deform + grid
            else:
                deformed_grid = grid
        else:
            pdb.set_trace()
            if encode_w:
                e_w = self.embeddings[f'w_{i}'](repeat(ts_w, 'b n ->  (b l) n ',
                                                    l=grid.shape[0])[:, 0])
                # Whether to use annealed positional encoding.
                if self.hparams.annealed: # False
                    pe_w = self.embeddings['xyz_w'][i](grid, step)
                else:
                    pe_w = self.embeddings['xyz_w'][i](grid)

                # Warping field type.
                deform = self.models[f'warping_field_{i}'](torch.cat(
                    [e_w, pe_w], 1))
                deformed_grid = deform + grid
            else:
                deformed_grid = grid

        # tensor [deformed_grid] size: [518400, 2], min: 0.024697, max: 1.060139, mean: 0.538712
        return deformed_grid

    def forward(self,
                ts_w,
                grid, # [1, 518400, 2]
                encode_w,
                step=0,
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
        flow_loss_list = []
        deform_list = []
        for i in range(self.num_models): # self.num_models === 1
            deformed_grid = self.deform_pts(ts_w, grid, encode_w, step, i)  # [batch * num_pixels, 2]
            deform_list.append(deformed_grid)
            # Compute optical flow loss.
            flow_loss = 0
            if self.hparams.flow_loss > 0 and not self.hparams.test: # False
                if flows.max() > -1e2 and step > self.hparams.flow_step:
                    grid_new = grid + flows.squeeze(0)
                    deformed_grid_new = self.deform_pts(
                        ts_w + 1, grid_new, encode_w, step, i)
                    flow_loss = (deformed_grid_new, deformed_grid)
            flow_loss_list.append(flow_loss)
            if self.hparams.vid_hash: # True
                pe_deformed_grid = (deformed_grid + 0.3) / 1.6
            else:
                pdb.set_trace()
                pe_deformed_grid = self.embeddings['xyz'](deformed_grid)
            if not self.training and self.hparams.canonical_dir is not None: # True
                w, h = self.img_wh
                canonical_img = self.canonical_img.squeeze(0)
                h_c, w_c = canonical_img.shape[1:3]
                grid_new = deformed_grid.clone()
                grid_new[..., 1] = (2 * deformed_grid[..., 0] - 1) * h / h_c
                grid_new[..., 0] = (2 * deformed_grid[..., 1] - 1) * w / w_c
                if len(canonical_img.shape) == 3:
                    canonical_img = canonical_img.unsqueeze(0)
                # ------------------------------------------------------------
                results = torch.nn.functional.grid_sample(
                    canonical_img[i:i + 1].permute(0, 3, 1, 2),
                    grid_new.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    padding_mode='border')
                # ------------------------------------------------------------
                # tensor [results] size: [518400, 3], min: 0.0, max: 0.996094, mean: 0.100389
                results = results.squeeze().permute(1,0)
            else:
                results = self.models[f'implicit_video_{i}'](pe_deformed_grid)

            results_list.append(results)

        ret = edict(rgbs=results_list,
                    flow_loss=flow_loss_list,
                    deform=deform_list)

        return ret

    def setup(self, stage):
        if not self.hparams.test: # False
            dataset = dataset_dict[self.hparams.dataset_name]
            kwargs = {
                'root_dir': self.hparams.root_dir,
                'img_wh': tuple(self.hparams.img_wh),
                'mask_dir': self.hparams.mask_dir,
                'flow_dir': self.hparams.flow_dir,
                'canonical_wh': self.hparams.canonical_wh,
                'ref_idx': self.hparams.ref_idx,
                'canonical_dir': self.hparams.canonical_dir
            }
            self.train_dataset = dataset(split='train', **kwargs)
            self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        lr_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [lr_dict]

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          sampler=sampler,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time.
            pin_memory=True)

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

    def training_step(self, batch, batch_idx):
        # Fetch training data.
        rgbs = batch['rgbs']
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        flows = batch['flows']
        grid_c = batch['grid_c']
        ref_batch = batch['reference']
        self.seq_len = batch['seq_len']

        loss = 0
        rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')

        # Forward the model.
        ret = self.forward(ts_w,
                            grid,
                            self.hparams.encode_w,
                            self.global_step,
                            flows=flows)

        # Mannually set a reference frame.
        if self.hparams.ref_step < 0: self.hparams.step = 1e10
        if (self.hparams.ref_idx is not None
                and self.global_step < self.hparams.ref_step):
            rgbs_c_flattend = rearrange(ref_batch[0],
                                        'b h w c -> (b h w) c')
            ret_c = self(ts_w, grid, False, self.global_step, flows=flows)

        # Loss computation.
        for i in range(self.num_models):
            results = ret.rgbs[i]
            mk_t = rearrange(mk[i], 'b h w c -> (b h w) c')
            mk_t = mk_t.sum(dim=-1) > 0.05

            if (self.hparams.ref_idx is not None
                and self.global_step < self.hparams.ref_step):
                mk_c_t = rearrange(ref_batch[1][i], 'b h w c -> (b h w) c')
                mk_c_t = mk_c_t.sum(dim=-1) > 0.05

            # Background regularization.
            if self.hparams.bg_loss:
                mk1 = torch.logical_not(mk_t)
                if self.hparams.self_bg:
                    grid_flattened = rgbs_flattend
                else:
                    grid_flattened = rearrange(grid, 'b n c -> (b n) c')
                    grid_flattened = torch.cat(
                        [grid_flattened, grid_flattened[:, :1]], -1)

            if self.hparams.bg_loss and self.hparams.mask_dir:
                loss = loss + self.hparams.bg_loss * self.color_loss(
                    results[mk1], grid_flattened[mk1])

            # MSE color loss.
            loss = loss + self.color_loss(results[mk_t],
                                            rgbs_flattend[mk_t])

            # Image gradient loss.
            img_pred = rearrange(results,
                                 '(b h w) c -> b h w c',
                                 b=1,
                                 h=self.h,
                                 w=self.w)
            rgbs_gt = rearrange(rgbs_flattend,
                                '(b h w) c -> b h w c',
                                b=1,
                                h=self.h,
                                w=self.w)
            mk_t_re = rearrange(mk_t,
                                '(b h w c) -> b h w c',
                                b=1,
                                h=self.h,
                                w=self.w)
            grad_loss = compute_gradient_loss(rgbs_gt.permute(0, 3, 1, 2),
                                              img_pred.permute(0, 3, 1, 2),
                                              mask=mk_t_re.permute(0, 3, 1, 2))
            loss = loss + grad_loss * self.hparams.grad_loss

            # Optical flow loss.
            if ret.flow_loss[0] != 0:
                mk_flow_t = torch.logical_and(mk_t, flows[0].sum(dim=-1)< 3)
                loss = loss + torch.nn.functional.l1_loss(
                    ret.flow_loss[i][0][mk_flow_t], ret.flow_loss[i][1]
                    [mk_flow_t]) * self.hparams.flow_loss

            # Reference loss.
            if (self.hparams.ref_idx is not None
                    and self.global_step < self.hparams.ref_step):
                results_c = ret_c.rgbs[i]
                loss += self.color_loss(results_c[mk_c_t],
                                        rgbs_c_flattend[mk_c_t])

            # PSNR metric.
            with torch.no_grad():
                if i == 0:
                    psnr_ = psnr(results[mk_t], rgbs_flattend[mk_t])

        self.log('lr', get_learning_rate(self.optimizer), prog_bar=True)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgbs = batch['rgbs']
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        grid_c = grid  # batch['grid_c']
        self.seq_len = batch['seq_len']
        ret = self(ts_w, grid, self.hparams.encode_w, self.global_step)
        ret_c = self(ts_w, grid_c, False, self.global_step)

        log = {}
        W, H = self.hparams.img_wh

        rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')
        img_gt = rgbs_flattend.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        stack_list = [img_gt]
        for i in range(self.num_models):
            results = ret.rgbs[i]
            results_c = ret_c.rgbs[i]
            mk_t = rearrange(mk[i], 'b h w c -> (b h w) c')
            if batch_idx == 0:
                results[mk_t.sum(dim=-1) <= 0.05] = 0
                img = results.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                img_c = results_c.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                stack_list.append(img)
                stack_list.append(img_c)

        stack = torch.stack(stack_list) # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_Reconstructed', stack,
                                          self.global_step)

        return log

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
        grid_c = batch['grid_c']
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
        if self.hparams.canonical_dir is not None: # True
            test_dir = f'{save_dir}_transformed'
            video_name = f'{sample_name}_{self.hparams.exp_name}_transformed'
        else:
            test_dir = f'{save_dir}'
            video_name = f'{sample_name}_{self.hparams.exp_name}'
        Path(test_dir).mkdir(parents=True, exist_ok=True)

        if batch_idx > 0 and self.hparams.save_video: # self.hparams.save_video -- True
            self.video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}.mp4'))
            self.raw_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_raw.mp4'))
            self.dual_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_dual.mp4'))

        if batch_idx == 0 and self.hparams.canonical_dir is None: # False
            pdb.set_trace()
            # Save the canonical image.
            ret = self.forward(ts_w, grid_c, False, self.global_step)
            pdb.set_trace()

        # --------------- forward ------------------------
        ret_n = self.forward(ts_w, grid, self.hparams.encode_w, self.global_step)
        # ret_n['rgbs'] is list: len = 1
        #     tensor [item] size: [518400, 3], min: 0.0, max: 0.996094, mean: 0.100389

        img = np.zeros((H * W, 3), dtype=np.float32) # (518400, 3), (H, W) === (540, 960)
        for i in range(self.num_models):
            if batch_idx == 0 and self.hparams.canonical_dir is None: # False
                results_c = ret.rgbs[i]
                if self.hparams.canonical_wh:
                    img_c = results_c.view(self.hparams.canonical_wh[1],
                                           self.hparams.canonical_wh[0],
                                           3).float().cpu().numpy()
                else:
                    img_c = results_c.view(H, W, 3).float().cpu().numpy()

                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'{test_dir}/canonical_{i}.png', img_c * 255)

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

        if self.hparams.save_deform: # False
            save_deform_dir = f'{test_dir}_deform'
            Path(save_deform_dir).mkdir(parents=True, exist_ok=True)
            deformation_field = ret_n.deform[0]
            deformation_field = rearrange(deformation_field,
                                          '(h w) c -> h w c', h=H, w=W)
            grid_ = rearrange(grid[0], '(h w) c -> h w c', h=H, w=W)
            deformation_delta = deformation_field - grid_
            np.save(f'{save_deform_dir}/{batch_idx:05d}.npy',
                    deformation_delta.cpu().numpy())

    def on_test_epoch_end(self):
        if self.hparams.save_video:
            self.video_visualizer.save()
            self.raw_video_visualizer.save()
            self.dual_video_visualizer.save()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

def main(hparams):
    system = ImplicitVideoSystem(hparams)

    if not hparams.test:
        os.makedirs(f'{hparams.model_save_path}/{hparams.exp_name}',
                    exist_ok=True)

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

    if hparams.test:
        trainer.test(system, dataloaders=system.test_dataloader())
    else:
        trainer.fit(system, ckpt_path=hparams.ckpt_path)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
