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
import numpy as np
import ground_based_dataset as gbd
import resnet_ssl_model as rsm

if sys.platform == 'linux':
    path = '/home/milesx/datasets/deeplens/'
    save_path = os.path.join(path, 'saved_model')
else:
    path = 'C:\\Users\\miles\\Documents\\dataset'
n_train_data = 512
n_eval_data = 128
batch_size = 128
num_classes = 2
num_epochs = 3
rampup_length = 80
rampdown_length = 50
learning_rate = 0.003
pred_decay = 0.6
embed = True
embed_coeff = 0.2
adam_beta1 = 0.9
rd_beta1_target = 0.5
augment_mirror = True
augment_translation = 2
unsup_wght = 0.0
whiten_inputs = 'norm'  # norm, zca
polyak_decay = 0.999
has_cuda = torch.cuda.is_available()
ground_train_dataset = gbd.GroundBasedDataset(path, length=n_train_data)
ground_train_loader = DataLoader(ground_train_dataset, batch_size=batch_size,
                                 shuffle=True, pin_memory=not has_cuda)
ground_eval_dataset = gbd.GroundBasedDataset(path, offset=n_train_data,
                                             length=n_eval_data)
ground_eval_loader = DataLoader(ground_eval_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=not has_cuda)


ssl_lens_net = rsm.ResNetSSL([3, 3, 3, 3, 3])

#labeled_loss = nn.BCEWithLogitsLoss()
labeled_loss = nn.CrossEntropyLoss()
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


def whiten_norm(images):
    images -= torch.mean(images, (1, 2, 3), True)
    images /= torch.mean(images ** 2, (1, 2, 3), True) ** 0.5
    return images


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    ssl_lens_net.to(device)
else:
    device = torch.device('cpu')
ensemble_pred = torch.zeros((n_train_data, num_classes), device=device)
target_pred = torch.zeros((n_train_data, num_classes), device=device)
t_one = torch.ones(())
epoch_pred = t_one.new_empty(
    (n_train_data, num_classes), dtype=torch.float32, device=device)
epoch_mask = t_one.new_empty(
    (n_train_data), dtype=torch.float32, device=device)
epoch_loss = t_one.new_empty(
    (batch_size, 4), dtype=torch.float32, device=device)


for epoch in range(num_epochs):
    train_time = -time.time()
    epoch_pred.zero_()
    epoch_mask.zero_()
    # epoch_loss.zero_()
    adam_param_update(optimizer, epoch)

    for i, data in enumerate(ground_train_loader, 0):
        images, is_lens, mask, indices = data
        #images = images.to(cuda_device)
        #is_lens = is_lens.to(cuda_device)
        #mask = mask.to(cuda_device)
        #indices = indices.to(cuda_device)

        if whiten_inputs:
            if whiten_inputs == 'norm':
                images = whiten_norm(images)
            elif whiten_inputs == 'zca':
                from zca_norm import ZCA
                whitener = ZCA(x=images)
                images = whitener.apply(images)

        if augment_translation:
            p2d = tuple([augment_translation] * 4)
            images = F.pad(images, p2d, 'reflect')
            # print(images.size())
            crop, n_xl = augment_translation, 101
            for image in images:
                if augment_mirror and np.random.uniform() > 0.5:
                    # image = image[:, :, ::-1]
                    image = torch.flip(image, [2])
                ofs0 = np.random.randint(0, 2 * crop + 1)
                ofs1 = np.random.randint(0, 2 * crop + 1)
                image = image[:, ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl]

        targets = torch.index_select(target_pred, 0, indices)

        optimizer.zero_grad()

        outputs, h_x = ssl_lens_net(images)
        predicts = F.softmax(outputs, dim=1)

        # update for ensemble
        for k, j in enumerate(indices):
            epoch_pred[j] = predicts[k]
            epoch_mask[j] = 1.0

        # labeled loss
        labeled_mask = mask.eq(0)
        #print(f'labeled: {labeled_mask} and unlabeled: {mask}')
        loss = labeled_loss(outputs[labeled_mask], is_lens[labeled_mask])
        epoch_loss[i, 0] = loss.item()

        # unlabeled loss
        unlabeled_loss = torch.mean((predicts - targets)**2)
        epoch_loss[i, 1] = unlabeled_loss.item()
        loss += unlabeled_loss * unsup_wght

        # SNTG loss
        if embed:
            half = int(h_x.size()[0] // 2)
            eucd2 = torch.mean((h_x[:half] - h_x[half:])**2, dim=1)
            eucd = torch.sqrt(eucd2)
            target_hard = torch.argmax(targets, dim=1).int()
            merged_tar = torch.where(mask == 0, target_hard, is_lens.int())
            neighbor_bool = torch.eq(merged_tar[:half], merged_tar[half:])
            eucd_y = torch.where(eucd < 1.0, (1.0 - eucd) ** 2,
                                 torch.zeros_like(eucd))
            embed_losses = torch.where(neighbor_bool, eucd2, eucd_y)
            embed_loss = torch.mean(embed_losses)
            epoch_loss[i, 2] = embed_loss.item()
            loss += embed_loss * unsup_wght * embed_coeff
            epoch_loss[i, 3] = loss.item()
        loss.backward()
        optimizer.step()

        # print(loss, unlabeled_loss)

    ensemble_pred = pred_decay * ensemble_pred + (1 - pred_decay) * epoch_pred
    targets_pred = ensemble_pred / (1.0 - pred_decay ** (epoch + 1))
    loss_mean = torch.mean(epoch_loss, 0)
    print(f"epoch {epoch}, time cosumed: {time.time() + train_time}, "
          f"labeled loss: {loss_mean[0].item()}, "
          f"unlabeled loss: {loss_mean[1].item()}, "
          f"SNTG loss: {loss_mean[2].item()}, "
          f"total loss: {loss_mean[3].item()}")

if not os.path.isdir(save_path):
    os.mkdir(save_path)
file_name = 'ground_based' + \
    datetime.datetime.now().isoformat(
        '-',
        timespec='minutes').replace(':', '-') + '.pth'
torch.save(ssl_lens_net.state_dict(), os.path.join(save_path, file_name))
