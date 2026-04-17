import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from utils import save_feature_maps
from transformers import SegformerModel


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self, ver):
        super().__init__()

        # MiT-b0 backbone
        if ver == 'mitb0':
            self.backbone = SegformerModel.from_pretrained('nvidia/mit-b0')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # MiT-b1 backbone
        if ver == 'mitb1':
            self.backbone = SegformerModel.from_pretrained('nvidia/mit-b1')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # MiT-b2 backbone
        if ver == 'mitb2':
            self.backbone = SegformerModel.from_pretrained('nvidia/mit-b2')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        x = (img - self.mean) / self.std
        x = self.backbone(x, output_hidden_states=True).hidden_states
        s4 = x[0]
        s8 = x[1]
        s16 = x[2]
        s32 = x[3]
        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}


# decoding module
class Decoder(nn.Module):
    def __init__(self, ver):
        super().__init__()

        # MiT-b0 backbone
        if ver == 'mitb0':
            self.app_conv = Conv(160, 160, 1, 1, 0)
            self.mo_conv = Conv(160, 160, 1, 1, 0)
            self.conv1 = ConvRelu(256, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.conv2 = ConvRelu(160, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.conv3 = ConvRelu(64, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.conv4 = ConvRelu(32, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

        # MiT-b1 and MiT-b2 backbones
        else:
            self.app_conv = Conv(320, 320, 1, 1, 0)
            self.mo_conv = Conv(320, 320, 1, 1, 0)
            self.conv1 = ConvRelu(512, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.conv2 = ConvRelu(320, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.conv3 = ConvRelu(128, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.conv4 = ConvRelu(64, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

    def forward(self, app_feats, mo_feats, recon):
        
        b, _, _ = recon.shape
        
        # Align Shape for Reconstructed Feature Map 
        app = recon[:, :1024, :].permute(0, 2, 1).view(b, 320, 32, 32)
        mo = recon[:, 1024:, :].permute(0, 2, 1).view(b, 320, 32, 32)
        
        x = self.conv1(app_feats['s32'] + mo_feats['s32'])
        x = self.cbam1(self.blend1(x))
        s16 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv2(self.app_conv(app) + self.mo_conv(mo)), s16], dim=1)
        x = self.cbam2(self.blend2(x))
        s8 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv3(app_feats['s8'] + mo_feats['s8']), s8], dim=1)
        x = self.cbam3(self.blend3(x))
        s4 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv4(app_feats['s4'] + mo_feats['s4']), s4], dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        score = F.interpolate(x, scale_factor=4, mode='bicubic')
        
        return score


# VOS model
class VOS(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.app_encoder = Encoder(ver)
        self.mo_encoder = Encoder(ver)
        self.modulator = Modulator()
        self.decoder = Decoder(ver)


# CMTM model
class CMTM(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.vos = VOS(ver)

    def forward(self, imgs, flows, index=None, epoch=None):
        
        B, L, _, H1, W1 = imgs.size()
        _, _, _, H2, W2 = flows.size()

        # Resize to 512p
        s = 512
        imgs = F.interpolate(imgs.view(B * L, -1, H1, W1), size=(s, s), mode='bicubic').view(B, L, -1, s, s)
        flows = F.interpolate(flows.view(B * L, -1, H2, W2), size=(s, s), mode='bicubic').view(B, L, -1, s, s)

        # For each frame
        score_lst = []
        mask_lst = []    
        
        for i in range(L):

            # Query frame prediction
            app_feats = self.vos.app_encoder(imgs[:, i])
            mo_feats = self.vos.mo_encoder(flows[:, i])
            
            # Input Token for CMTM
            app_token = app_feats['s16'].flatten(2).permute(0, 2, 1) 
            mo_token = mo_feats['s16'].flatten(2).permute(0, 2, 1)
            
            # Domain Adaptive Masked AutoEncoder
            masked_x, recon = self.vos.modulator(app_token, mo_token)
            
            # Prediction Mask 
            score = self.vos.decoder(app_feats, mo_feats, recon)
            score = F.interpolate(score, size=(H1, W1), mode='bicubic')
            
            # Visualization the last of epoch 
            if index == 31:
                save_feature_maps(imgs[:, i], 
                                  flows[:, i], 
                                  app_token.detach(), 
                                  mo_token.detach(),
                                  masked_x.detach(), 
                                  recon.detach(), 
                                  epoch)

            # Store soft scores
            if B != 1:
                score_lst.append(score)

            # Store hard masks
            if B == 1:
                pred_seg = torch.softmax(score, dim=1)
                pred_mask = torch.max(pred_seg, dim=1, keepdim=True)[1]
                mask_lst.append(pred_mask)

        # Generate output
        output = {}
        
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
            
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
            
        return output
