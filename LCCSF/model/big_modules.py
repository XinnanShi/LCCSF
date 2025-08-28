"""
big_modules.py - This file stores higher-level network blocks.

x - usually denotes features that are shared between objects.
g - usually denotes features that are not shared between objects 
    with an extra "num_objects" dimension (batch_size * num_objects * num_channels * H * W).

The trailing number of a variable usually denotes the stride
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model.utils import resnet
from model.modules import *

import math

import numbers
from einops import rearrange
import torch_dct as dct

from timm.layers import to_2tuple
import numpy as np


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import uuid
import time

class FeatureSaver:
    def __init__(self, base_dir="feature_maps", sub_dir_format="%Y%m%d_%H%M%S"):
        self.base_dir = base_dir
        self.sub_dir_format = sub_dir_format
        self.current_dir = None
        self.frame_count = 0
        os.makedirs(base_dir, exist_ok=True)
        
    def create_new_session(self):
        timestamp = datetime.now().strftime(self.sub_dir_format)
        self.current_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        os.makedirs(self.current_dir, exist_ok=True)
        self.frame_count = 0
        print(f"{self.current_dir}")
        
    def save_feature_map(self, feature, name, normalize=True):
        if self.current_dir is None:
            self.create_new_session()
            
        if feature.is_cuda:
            feature = feature.cpu()
        feature_np = feature.detach().numpy()
        
        if feature_np.ndim == 3 and feature_np.shape[0] > 1:
            feature_np = np.mean(feature_np, axis=0)
        elif feature_np.ndim == 3 and feature_np.shape[0] == 1:
            feature_np = feature_np[0]
        current_timestamp = int(time.time())
        if normalize:
            vmin, vmax = np.min(feature_np), np.max(feature_np)
            if vmax - vmin > 1e-8:
                feature_np = (feature_np - vmin) / (vmax - vmin)
            else:
                feature_np = np.zeros_like(feature_np)
        
        feature_np = (feature_np * 255).astype(np.uint8)
        colored = cv2.applyColorMap(feature_np, cv2.COLORMAP_JET)
        
        frame_id = f"{self.frame_count:06d}"
        filename = f"{name}_{frame_id}_{current_timestamp}.png"
        save_path = os.path.join(self.current_dir, filename)
        
        cv2.imwrite(save_path, colored)
        return save_path

class LFFT(nn.Module):
    def __init__(self, dim, patch_size, ffn_expansion_factor=4, bias=True):
        super(LFFT, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = patch_size
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        #print(x.shape)
        #1,256,180,340
        #16,256,120,120
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)
        return x
        
        
        
class HCD(nn.Module):
    def __init__(self, dim=16, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),  
            act_layer() 
        )
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), 
            act_layer()  
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, dim)  
        )
        self.dim = dim  
        self.hidden_dim = hidden_dim  

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False) 

    def forward(self, x, H, W):
        bs, hw, c = x.size() 
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=H, w=W)  
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)  
        x1 = self.partial_conv3(x1)  
        x = torch.cat((x1, x2), 1) 
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W) 
        x = self.linear1(x)  
        x_1, x_2 = x.chunk(2, dim=-1)  
        x_1 = rearrange(x_1, 'b (h w) (c) -> b c h w', h=H, w=W)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c', h=H, w=W)  
        x = x_1 * x_2 
        x = self.linear2(x) 

        return x  

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class PixelEncoder(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.feature_saver = FeatureSaver(base_dir="davis_features")
        self.save_features = True 
        self.is_resnet = 'resnet' in model_cfg.pixel_encoder.type
        resnet_model_path = model_cfg.get('resnet_model_path')
        #self.FCA = FCAttention(channel = 256)
        self.HCD = HCD(dim = 256)
        self.LFFT = LFFT(dim = 256, patch_size = 4)
        self.a = torch.nn.Parameter(torch.tensor(1.005))
        self.b = torch.nn.Parameter(torch.tensor(1.0))
        self.c = torch.nn.Parameter(torch.tensor(1.0))
        #self.block = FEFM(dim = 256, bias = False, depth = 1)
        #self.ENA = ImageProcessingModule()
        #self.FUS = FusionConv(in_channels=512, out_channels=256)
        if self.is_resnet:
            if model_cfg.pixel_encoder.type == 'resnet18':
                network = resnet.resnet18(pretrained=True, model_dir=resnet_model_path)
            elif model_cfg.pixel_encoder.type == 'resnet50':
                network = resnet.resnet50(pretrained=True, model_dir=resnet_model_path)
            else:
                raise NotImplementedError
            self.conv1 = network.conv1
            self.bn1 = network.bn1
            self.relu = network.relu
            self.maxpool = network.maxpool

            self.res2 = network.layer1
            self.layer2 = network.layer2
            self.layer3 = network.layer3
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        #print(self.a,self.b,self.c)
        #x1 = x
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.maxpool(x)
        #print(x.shape)
        f4 = self.res2(x)
        
        #print(f4.shape,to_3d(f4).shape)
        #f41 = self.FCA(f4)
        '''
        x1 = self.ENA(f4, x1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        f43 = self.res2(x1)
        H, W = f43.shape[2], f43.shape[3]
        f43 = to_4d(self.FRFN(to_3d(f4), H, W), H, W)
        '''
        f40 = f4
        f41 = self.LFFT(f4)
        H, W = f4.shape[2], f4.shape[3]
        f42 = to_4d(self.HCD(to_3d(f4), H, W), H, W)
        #print(self.a,self.b,self.c)
        f4 = self.a * f4 + self.b * f41 + self.c * f42
        #print(f4.shape)
        #print('f4',f4.shape)
        ''' 
        16,64,240,240
        torch.Size([32, 64, 120, 120])
        f4 torch.Size([32, 256, 120, 120])
        '''
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        '''
        if self.save_features:
            self.feature_saver.save_feature_map(f40[0], "f40")  
            self.feature_saver.save_feature_map(f41[0], "f41")
            self.feature_saver.save_feature_map(f42[0], "f42")
            self.feature_saver.save_feature_map(f4[0], "f4_fused")
            self.feature_saver.save_feature_map(f8[0], "f8_fused")
            self.feature_saver.save_feature_map(f16[0], "f16_fused")
        '''
        #f4 = self.block(f40,  (f41 + f42))
       
        
        return f16, f8, f4

    # override the default train() to freeze BN statistics
    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class KeyProjection(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        in_dim = model_cfg.pixel_encoder.ms_dims[0]
        mid_dim = model_cfg.pixel_dim
        key_dim = model_cfg.key_dim

        self.pix_feat_proj = nn.Conv2d(in_dim, mid_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(mid_dim, key_dim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(mid_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(mid_dim, key_dim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x: torch.Tensor, *, need_s: bool,
                need_e: bool) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.pix_feat_proj(x)
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class MaskEncoder(nn.Module):
    def __init__(self, model_cfg: DictConfig, single_object=False):
        super().__init__()
        pixel_dim = model_cfg.pixel_dim
        value_dim = model_cfg.value_dim
        sensory_dim = model_cfg.sensory_dim
        final_dim = model_cfg.mask_encoder.final_dim

        self.single_object = single_object
        extra_dim = 1 if single_object else 2

        resnet_model_path = model_cfg.get('resnet_model_path')
        if model_cfg.mask_encoder.type == 'resnet18':
            network = resnet.resnet18(pretrained=True, extra_dim=extra_dim, model_dir=resnet_model_path)
        elif model_cfg.mask_encoder.type == 'resnet50':
            network = resnet.resnet50(pretrained=True, extra_dim=extra_dim, model_dir=resnet_model_path)
        else:
            raise NotImplementedError
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool

        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

        self.distributor = MainToGroupDistributor()
        self.fuser = GroupFeatureFusionBlock(pixel_dim, final_dim, value_dim)

        self.sensory_update = SensoryDeepUpdater(value_dim, sensory_dim)

    def forward(self,
                image: torch.Tensor,
                pix_feat: torch.Tensor,
                sensory: torch.Tensor,
                masks: torch.Tensor,
                others: torch.Tensor,
                *,
                deep_update: bool = True,
                chunk_size: int = -1) -> (torch.Tensor, torch.Tensor):
        # ms_features are from the key encoder
        # we only use the first one (lowest resolution), following XMem
        if self.single_object:
            g = masks.unsqueeze(2)
        else:
            g = torch.stack([masks, others], dim=2)

        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            if deep_update:
                new_sensory = torch.empty_like(sensory)
            else:
                new_sensory = sensory
            fast_path = False

        # chunk-by-chunk inference
        all_g = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                g_chunk = g
            else:
                g_chunk = g[:, i:i + chunk_size]
            actual_chunk_size = g_chunk.shape[1]
            g_chunk = g_chunk.flatten(start_dim=0, end_dim=1)

            g_chunk = self.conv1(g_chunk)
            g_chunk = self.bn1(g_chunk)  # 1/2, 64
            g_chunk = self.maxpool(g_chunk)  # 1/4, 64
            g_chunk = self.relu(g_chunk)

            g_chunk = self.layer1(g_chunk)  # 1/4
            g_chunk = self.layer2(g_chunk)  # 1/8
            g_chunk = self.layer3(g_chunk)  # 1/16

            g_chunk = g_chunk.view(batch_size, actual_chunk_size, *g_chunk.shape[1:])
            g_chunk = self.fuser(pix_feat, g_chunk)
            all_g.append(g_chunk)
            if deep_update:
                if fast_path:
                    new_sensory = self.sensory_update(g_chunk, sensory)
                else:
                    new_sensory[:, i:i + chunk_size] = self.sensory_update(
                        g_chunk, sensory[:, i:i + chunk_size])
        g = torch.cat(all_g, dim=1)

        return g, new_sensory

    # override the default train() to freeze BN statistics
    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class PixelFeatureFuser(nn.Module):
    def __init__(self, model_cfg: DictConfig, single_object=False):
        super().__init__()
        value_dim = model_cfg.value_dim
        sensory_dim = model_cfg.sensory_dim
        pixel_dim = model_cfg.pixel_dim
        embed_dim = model_cfg.embed_dim
        self.single_object = single_object

        self.fuser = GroupFeatureFusionBlock(pixel_dim, value_dim, embed_dim)
        if self.single_object:
            self.sensory_compress = GConv2d(sensory_dim + 1, value_dim, kernel_size=1)
        else:
            self.sensory_compress = GConv2d(sensory_dim + 2, value_dim, kernel_size=1)

    def forward(self,
                pix_feat: torch.Tensor,
                pixel_memory: torch.Tensor,
                sensory_memory: torch.Tensor,
                last_mask: torch.Tensor,
                last_others: torch.Tensor,
                *,
                chunk_size: int = -1) -> torch.Tensor:
        batch_size, num_objects = pixel_memory.shape[:2]

        if self.single_object:
            last_mask = last_mask.unsqueeze(2)
        else:
            last_mask = torch.stack([last_mask, last_others], dim=2)

        if chunk_size < 1:
            chunk_size = num_objects

        # chunk-by-chunk inference
        all_p16 = []
        for i in range(0, num_objects, chunk_size):
            sensory_readout = self.sensory_compress(
                torch.cat([sensory_memory[:, i:i + chunk_size], last_mask[:, i:i + chunk_size]], 2))
            p16 = pixel_memory[:, i:i + chunk_size] + sensory_readout
            p16 = self.fuser(pix_feat, p16)
            all_p16.append(p16)
        p16 = torch.cat(all_p16, dim=1)

        return p16


class MaskDecoder(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        embed_dim = model_cfg.embed_dim
        sensory_dim = model_cfg.sensory_dim
        ms_image_dims = model_cfg.pixel_encoder.ms_dims
        up_dims = model_cfg.mask_decoder.up_dims

        assert embed_dim == up_dims[0]

        self.sensory_update = SensoryUpdater([up_dims[0], up_dims[1], up_dims[2] + 1], sensory_dim,
                                             sensory_dim)

        self.decoder_feat_proc = DecoderFeatureProcessor(ms_image_dims[1:], up_dims[:-1])
        self.up_16_8 = MaskUpsampleBlock(up_dims[0], up_dims[1])
        self.up_8_4 = MaskUpsampleBlock(up_dims[1], up_dims[2])

        self.pred = nn.Conv2d(up_dims[-1], 1, kernel_size=3, padding=1)

    def forward(self,
                ms_image_feat: Iterable[torch.Tensor],
                memory_readout: torch.Tensor,
                sensory: torch.Tensor,
                *,
                chunk_size: int = -1,
                update_sensory: bool = True) -> (torch.Tensor, torch.Tensor):

        batch_size, num_objects = memory_readout.shape[:2]
        f8, f4 = self.decoder_feat_proc(ms_image_feat[1:])
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            if update_sensory:
                new_sensory = torch.empty_like(sensory)
            else:
                new_sensory = sensory
            fast_path = False

        # chunk-by-chunk inference
        all_logits = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                p16 = memory_readout
            else:
                p16 = memory_readout[:, i:i + chunk_size]
            actual_chunk_size = p16.shape[1]

            p8 = self.up_16_8(p16, f8)
            p4 = self.up_8_4(p8, f4)
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.pred(F.relu(p4.flatten(start_dim=0, end_dim=1).float()))

            if update_sensory:
                p4 = torch.cat(
                    [p4, logits.view(batch_size, actual_chunk_size, 1, *logits.shape[-2:])], 2)
                if fast_path:
                    new_sensory = self.sensory_update([p16, p8, p4], sensory)
                else:
                    new_sensory[:,
                                i:i + chunk_size] = self.sensory_update([p16, p8, p4],
                                                                        sensory[:,
                                                                                i:i + chunk_size])
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return new_sensory, logits
