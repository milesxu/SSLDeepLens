import sys
import math
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
else:
    path = 'C:\\Users\\miles\\Documents\\dataset'
n_data = 20
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
ground_based_dataset = gbd.GroundBasedDataset(path, length=n_data)
ground_train_loader = DataLoader(ground_based_dataset, batch_size=10,
                                 shuffle=True)
ensemble_pred = torch.zeros((n_data, num_classes))
target_pred = torch.zeros((n_data, num_classes))
unsup_wght = 0.0

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


for epoch in range(num_epochs):

    epoch_pred = torch.zeros((n_data, num_classes))
    epoch_mask = torch.zeros((n_data))

    adam_param_update(optimizer, epoch)

    for i, data in enumerate(ground_train_loader, 0):
        images, is_lens, mask, indices = data

        if augment_translation:
            p2d = tuple([augment_translation] * 4)
            images = F.pad(images, p2d, 'reflect')
            print(images.size())
            crop, n_xl = augment_translation, 101
            for image in images:
                if augment_mirror and np.random.uniform() > 0.5:
                    # image = image[:, :, ::-1]
                    image = torch.flip(image, [2])
                ofs0 = np.random.randint(0, 2 * crop + 1)
                ofs1 = np.random.randint(0, 2 * crop + 1)
                image = image[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]

        targets = torch.index_select(target_pred, 0, indices)

        optimizer.zero_grad()

        outputs, h_x = ssl_lens_net(images)
        predicts = F.softmax(outputs, dim=1)

        # update for ensemble
        for i, j in enumerate(indices):
            epoch_pred[j] = predicts[i]
            epoch_mask[j] = 1.0

        # labeled loss
        labeled_mask = mask.eq(0)
        #print(f'labeled: {labeled_mask} and unlabeled: {mask}')
        loss = labeled_loss(outputs[labeled_mask], is_lens[labeled_mask])

        # unlabeled loss
        unlabeled_loss = torch.mean((predicts - targets)**2)
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
            loss += embed_loss * unsup_wght * embed_coeff
        loss.backward()
        optimizer.step()

        print(loss.item())

    ensemble_pred = pred_decay * ensemble_pred + (1 - pred_decay) * epoch_pred
    targets_pred = ensemble_pred / (1.0 - pred_decay ** (epoch + 1))
