import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import numpy as np
import ground_based_dataset as gbd
import resnet_ssl_model as rsm

# path = 'C:\\Users\\miles\\Documents\\dataset'
path = '/home/milesx/datasets/deeplens/'
n_data = 20
num_classes = 2
pred_decay = 0.6
embed = True
embed_coeff = 0.2
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

for epoch in range(2):
    epoch_pred = torch.zeros((n_data, num_classes))
    epoch_mask = torch.zeros((n_data))
    for i, data in enumerate(ground_train_loader, 0):
        images, is_lens, mask, indices = data
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
            print(h_x.size(), half, h_x[:half].size(), h_x[half:].size())
            print(h_x[:half]-h_x[half:])
            eucd2 = torch.mean((h_x[:half] - h_x[half:])**2, dim=1)
            print('eucd2 ', eucd2)
            eucd = torch.sqrt(eucd2)
            print(eucd2.size(), eucd.size())
            target_hard = torch.argmax(targets, dim=1).int()
            print(target_hard, target_hard.size())
            merged_tar = torch.where(
                mask.int() == 0, target_hard, is_lens.int())
            print(merged_tar, merged_tar.size())
            neighbor_bool = torch.eq(merged_tar[:half], merged_tar[half:])
            print(neighbor_bool)
            eucd_y = torch.where(eucd < 1.0, (1.0 - eucd)
                                 ** 2, torch.zeros_like(eucd))
            print(eucd_y)
            embed_losses = torch.where(neighbor_bool, eucd2, eucd_y)
            print(embed_losses)
            embed_loss = torch.mean(embed_losses)
            print(embed_loss)
            loss += embed_loss * unsup_wght * embed_coeff
        loss.backward()
        optimizer.step()

        print(loss.item())

    ensemble_pred = pred_decay * ensemble_pred + (1 - pred_decay) * epoch_pred
    targets_pred = ensemble_pred / (1.0 - pred_decay ** (epoch + 1.0))
