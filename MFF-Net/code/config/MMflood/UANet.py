import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.flood_dataset import *
from geoseg.models.UANet_f import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def dice_loss(self, predictions, targets):
        smooth = 1e-6
        mask = (targets != self.ignore_index).unsqueeze(1)  # [N, 1, H, W]
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot * mask.float()
        predictions = F.softmax(predictions, dim=1) * mask.float()
        intersection = torch.sum(predictions * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(predictions, dim=(0, 2, 3)) + torch.sum(targets_one_hot, dim=(0, 2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_score.mean()

    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


max_epoch = 101
ignore_index = 255
train_batch_size = 12
val_batch_size = 12
lr = 1e-3
weight_decay = 0.0025
backbone_lr = 1e-3
backbone_weight_decay = 0.0025
accumulate_n = 1
num_classes = len(CLASSES) 
classes = CLASSES  
weights_name = "UANet_Res250"
weights_path = "model_weights/MMflood/{}".format(weights_name)
test_weights_name = "UANet_Res250"
log_name = 'MMflood/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = './UANet_Res250.ckpt'
resume_ckpt_path = None

net = UANet_Res250(channel=32, num_classes=num_classes)


loss = CombinedLoss(ignore_index=ignore_index,num_classes=num_classes)
use_aux_loss = False

data_root = '/root/autodl-tmp/mm_processed'

train_dataset = SARBuildingDataset(
    data_root=os.path.join(data_root, 'train'),
    mode='train',
    img_dir='sar',
    mask_dir='mask',
    img_suffix='.tif',
    mask_suffix='.tif',
    transform=train_aug,
    mosaic_ratio=0.25,
    img_size=(1024, 1024)
)

val_dataset = SARBuildingDataset(
    data_root=os.path.join(data_root, 'val'),
    mode='val',
    img_dir='sar',
    mask_dir='mask',
    img_suffix='.tif',
    mask_suffix='.tif',
    transform=val_aug,
    mosaic_ratio=0.0,
    img_size=(1024, 1024)
)

test_dataset = SARBuildingDataset(
    data_root=os.path.join(data_root, 'test'),
    mode='test',
    img_dir='sar',
    mask_dir='mask',
    img_suffix='.tif',
    mask_suffix='.tif',
    transform=val_aug,
    mosaic_ratio=0.0,
    img_size=(1024, 1024)
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
