import os
import png
import time
import math
import numpy
import queue
import threading

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F




DAVIS_PALETTE_4BIT = [[  0,   0,   0],
                      [128,   0,   0],
                      [  0, 128,   0],
                      [128, 128,   0],
                      [  0,   0, 128],
                      [128,   0, 128],
                      [  0, 128, 128],
                      [128, 128, 128],
                      [ 64,   0,   0],
                      [191,   0,   0],
                      [ 64, 128,   0],
                      [191, 128,   0],
                      [ 64,   0, 128],
                      [191,   0, 128],
                      [ 64, 128, 128],
                      [191, 128, 128]]


class ReadSaveImage(object):
    def __init__(self):
        super().__init__()

    def check_path(self, fullpath):
        path, filename = os.path.split(fullpath)
        if not os.path.exists(path):
            os.makedirs(path)


class DAVISLabels(ReadSaveImage):
    def __init__(self):
        super().__init__()
        self._width = 0
        self._height = 0

    def save(self, image, path):
        self.check_path(path)
        bitdepth = int(math.log(len(DAVIS_PALETTE_4BIT)) / math.log(2))
        height, width = image.shape
        file = open(path, 'wb')
        writer = png.Writer(width, height, palette=DAVIS_PALETTE_4BIT, bitdepth=bitdepth)
        writer.write(file, image)

    def read(self, path):
        try:
            reader = png.Reader(path)
            width, height, data, meta = reader.read()
            image = numpy.vstack(data)
            self._height, self._width = image.shape
        except png.FormatError:
            image = numpy.zeros((self._height, self._width))
            self.save(image, path)
        return image


class ImageSaver(threading.Thread):
    def __init__(self):
        super().__init__()
        self._alive = True
        self._queue = queue.Queue(2 ** 20)
        self.start()

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, alive):
        self._alive = alive

    @property
    def queue(self):
        return self._queue

    def kill(self):
        self._alive = False

    def enqueue(self, datatuple):
        ret = True
        try:
            self._queue.put(datatuple, block=False)
        except queue.Full:
            print('enqueue full')
            ret = False
        return ret

    def run(self):
        while True:
            while not self._queue.empty():
                args, method = self._queue.get(block=False, timeout=2)
                method.save(*args)
                self._queue.task_done()
            if not self._alive and self._queue.empty():
                break
            time.sleep(0.0001)


class AverageMeter(object):
    def __init__(self):
        self.clear()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 'nan'

    def new_epoch(self):
        self.history.append(self.avg)
        self.reset()


def get_iou(predictions, gt):
    nsamples, nclasses, height, width = predictions.size()
    prediction_max, prediction_argmax = predictions.max(-3)
    prediction_argmax = prediction_argmax.long()
    classes = gt.new_tensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1)
    pred_bin = (prediction_argmax.view(nsamples, 1, height, width) == classes)
    gt_bin = (gt.view(nsamples, 1, height, width) == classes)
    intersection = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=-2).sum(dim=-1)
    return (intersection + 1e-7) / (union + 1e-7)


def save_feature_maps(img, flow, app, mo, masked, recon, epoch):
        
        # img: B x 3 x H x W
        # flow: B x 3 x H x W
        # target: B x 2N x C 
        # masked: B x 2N x C 
        # recon: B x 2N x C 
        
        os.makedirs('./feature_map', exist_ok=True)
        
        # Original Image
        img = img[0:1]
        flow = flow[0:1]
        
        # Original Feature 
        app_feat = app[0:1]
        mo_feat = mo[0:1]
        
        # Masked Token 
        masked = masked[0:1]
        app_masked = masked[:, :1024, :]
        mo_masked = masked[:, 1024:, :]

        # Reconstucted Token
        recon = recon[0:1]
        app_recon = recon[:, :1024, :]
        mo_recon = recon[:, 1024:, :]
        
        # Original Feature  
        app_feat = app_feat.transpose(1, 2).view(1, 320, 32, 32)
        x = torch.mean(app_feat ** 2, dim=1, keepdim=True).view(1, 1, 32, 32)
        x[0, 0, 0, 0] = torch.mean(x)
        x_max = torch.max(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x_min = torch.min(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        x1 = F.interpolate(x.repeat(1, 3, 1, 1), scale_factor=4, mode='nearest')
        
        mo_feat = mo_feat.transpose(1, 2).view(1, 320, 32, 32)
        x = torch.mean(mo_feat ** 2, dim=1, keepdim=True).view(1, 1, 32, 32)
        x[0, 0, 0, 0] = torch.mean(x)
        x_max = torch.max(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x_min = torch.min(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        x2 = F.interpolate(x.repeat(1, 3, 1, 1), scale_factor=4, mode='nearest')
 
        # Masked Image 
        app_masked = app_masked.transpose(1, 2).view(1, 320, 32, 32)
        x = torch.mean(app_masked ** 2, dim=1, keepdim=True).view(1, 1, 32, 32)
        x[0, 0, 0, 0] = torch.mean(x)
        x_max = torch.max(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x_min = torch.min(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        x5 = F.interpolate(x.repeat(1, 3, 1, 1), scale_factor=4, mode='nearest')
        
        mo_masked = mo_masked.transpose(1, 2).view(1, 320, 32, 32)
        x = torch.mean(mo_masked ** 2, dim=1, keepdim=True).view(1, 1, 32, 32)
        x[0, 0, 0, 0] = torch.mean(x)
        x_max = torch.max(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x_min = torch.min(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        x6 = F.interpolate(x.repeat(1, 3, 1, 1), scale_factor=4, mode='nearest')
        
        # Reconstructed Image
        app_recon = app_recon.transpose(1, 2).view(1, 320, 32, 32)
        x = torch.mean(app_recon ** 2, dim=1, keepdim=True).view(1, 1, 32, 32)
        x[0, 0, 0, 0] = torch.mean(x)
        x_max = torch.max(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x_min = torch.min(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        x7 = F.interpolate(x.repeat(1, 3, 1, 1), scale_factor=4, mode='nearest')
        
        mo_recon = mo_recon.transpose(1, 2).view(1, 320, 32, 32)
        x = torch.mean(mo_recon ** 2, dim=1, keepdim=True).view(1, 1, 32, 32)
        x[0, 0, 0, 0] = torch.mean(x)
        x_max = torch.max(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x_min = torch.min(x.view(1, -1), dim=1, keepdim=True)[0].view(1, 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        x8 = F.interpolate(x.repeat(1, 3, 1, 1), scale_factor=4, mode='nearest')
        
        x = torch.cat([F.avg_pool2d(img, 4), x1, x5, x7,
                       F.avg_pool2d(flow, 4), x2, x6, x8], dim=0)
        
        os.makedirs('feature_map', exist_ok=True)
        
        tv.utils.save_image(x, 'feature_map/out_{:04d}.jpg'.format(epoch), nrow=4)