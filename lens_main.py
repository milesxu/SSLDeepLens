import os
import sys
import math
import datetime
import time
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
import ground_based_dataset as gbd
import resnet_ssl_model as rsm
from data_transforms import Clamp, AugmentTranslate, BatchPadding, WhitenInput
from ema import EMA

if sys.platform == 'linux':
    path = '/home/milesx/datasets/deeplens/'
else:
    path = 'C:\\Users\\miles\\Documents\\dataset'
save_path = os.path.join(path, 'saved_model')
train_params = {
    'n_data' : 1024,
    'num_classes' : 2,
    'batch_size' : 128,
    'n_eval_data' : 128,
    'test_offset' : 10000,
    'test_len' : 1000,
    'run_eval' : True,
    'run_test' : False,
    'num_epochs' : 9,
    'rampup_length' : 80,
    'rampdown_length' : 50,
    'learning_rate' : 0.003,
    'pred_decay' : 0.6,
    'embed' : True,
    'embed_coeff' : 0.2,
    'adam_beta1' : 0.9,
    'rd_beta1_target' : 0.5,
    'augment_mirror' : True,
    'augment_translation' : 2,
    'unsup_wght' : 0.0,
    'whiten_inputs' : 'norm',  # norm, zca
    'polyak_decay' : 0.999
}
torch.manual_seed(770715)
has_cuda = torch.cuda.is_available()
if has_cuda:
    torch.cuda.manual_seed_all(770715)
    torch.backends.cudnn.deterministic=True
composed = transforms.Compose(
    [Clamp(1e-9, 100), WhitenInput(),
     AugmentTranslate(augment_translation, 101)])
batch_composed = transforms.Compose([BatchPadding(augment_translation)])
ground_train_dataset = gbd.GroundBasedDataset(path, length=n_train_data,
                                              transform=composed)
                                            #   batch_transform=batch_composed)
ground_train_loader = DataLoader(ground_train_dataset, batch_size=batch_size,
                                 shuffle=True, pin_memory=not has_cuda)
ground_eval_dataset = gbd.GroundBasedDataset(path, offset=n_train_data,
                                             length=n_eval_data,
                                             transform=composed)
                                            #  batch_transform=batch_composed)
ground_eval_loader = DataLoader(ground_eval_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=not has_cuda)
ground_test_dataset = gbd.GroundBasedDataset(path, offset=test_offset,
                                             length=test_len,
                                             transform=composed)
                                            #  batch_transform=batch_composed)
ground_test_loader = DataLoader(ground_test_dataset, pin_memory=not has_cuda)


ssl_lens_net = rsm.ResNetSSL([3, 3, 3, 3, 3])
ema = EMA(polyak_decay, ssl_lens_net, has_cuda=has_cuda)

#labeled_loss = nn.BCEWithLogitsLoss()
optimizer = opt.Adam(ssl_lens_net.parameters())


def rampup(epoch):
    if epoch < rampup_length:
        p = 1.0 - epoch / rampup_length
        return math.exp(-p * p * 5.0)
    return 1.0


def rampdown(epoch):
    if epoch >= num_epochs - rampdown_length:
        ep = (epoch - num_epochs + rampdown_length) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0


def adam_param_update(optimizer: opt.Adam, epoch):
    ru, rd = rampup(epoch), rampdown(epoch)
    for group in optimizer.param_groups:
        group['lr'] = ru * rd * learning_rate
        group['betas'] = (rd * adam_beta1 + (1.0 - rd) * rd_beta1_target,
                          group['betas'][1])


# def whiten_norm(images):
#     images -= torch.mean(images, (1, 2, 3), True)
#     images /= torch.mean(images ** 2, (1, 2, 3), True) ** 0.5
#     return images


if has_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')




if not os.path.isdir(save_path):
    os.mkdir(save_path)
file_name = 'ground_based' + \
    datetime.datetime.now().isoformat(
        '-',
        timespec='minutes').replace(':', '-') + '.pth'
torch.save(ssl_lens_net.state_dict(), os.path.join(save_path, file_name))

if run_test:
    test_net = rsm.ResNetSSL([3, 3, 3, 3, 3])
    test_net.load_state_dict(torch.load(os.path.join(save_path, file_name)))
    test_net.eval()
