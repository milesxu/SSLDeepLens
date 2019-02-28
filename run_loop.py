import datetime
import time
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from ema import EMA


class SNTGRunLoop(object):
    def __init__(self, net, dataloader, params, update_fn, eval_loader=None,
                 test_loader=None, has_cuda=True):
        if has_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device('cpu')
        self.net = net
        self.loader = dataloader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.params = params
        self.device = device
        self.net.to(device)
        n_data, num_classes = params['n_data'], params['num_classes']
        self.ensemble_pred = torch.zeros((n_data, num_classes), device=device)
        self.target_pred = torch.zeros((n_data, num_classes), device=device)
        t_one = torch.ones(())
        self.epoch_pred = t_one.new_empty(
            (n_data, num_classes), dtype=torch.float32, device=device)
        self.epoch_mask = t_one.new_empty(
            (n_data), dtype=torch.float32, device=device)
        self.epoch_loss = t_one.new_empty((n_data // params['batch_size'], 4),
                                          dtype=torch.float32, device=device)
        self.optimizer = opt.Adam(self.net.parameters())
        self.update_fn = update_fn
        self.ema = EMA(params['polyak_decay'], self.net, has_cuda)
        # self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        self.net.train()
        #labeled_loss = nn.CrossEntropyLoss()
        train_losses, train_accs = [], []
        eval_losses, eval_accs = [], []
        for epoch in range(self.params['num_epochs']):
            # training phase
            train_time = -time.time()
            self.epoch_pred.zero_()
            self.epoch_mask.zero_()
            # epoch_loss.zero_()
            self.update_fn(self.optimizer, epoch)

            for i, data_batched in enumerate(self.loader, 0):
                images, is_lens, mask, indices = \
                    data_batched['image'], data_batched['is_lens'], \
                    data_batched['mask'], data_batched['index']

                targets = torch.index_select(self.target_pred, 0, indices)
                # print(f"y value dimension:{is_lens.size()}")

                self.optimizer.zero_grad()

                outputs, h_x = self.net(images)
                # print(f"output dimension: {outputs.size()}")
                predicts = F.softmax(outputs, dim=1)

                # update for ensemble
                for k, j in enumerate(indices):
                    self.epoch_pred[j] = predicts[k]
                    self.epoch_mask[j] = 1.0

                # labeled loss
                labeled_mask = mask.eq(0)
                # loss = self.loss_fn(
                # outputs[labeled_mask], is_lens[labeled_mask])
                # labeled loss with binary entropy with logits, use one_hot
                one_hot = torch.zeros(
                    len(is_lens[labeled_mask]), is_lens[labeled_mask].max()+1,
                    device=self.device) \
                    .scatter_(1, is_lens[labeled_mask].unsqueeze(1), 1.)
                loss = F.binary_cross_entropy_with_logits(outputs[labeled_mask],
                                                          one_hot)
                # print(loss.item())

                train_acc = torch.mean(torch.argmax(
                    outputs[labeled_mask], 1).eq(is_lens[labeled_mask]).float())
                train_accs.append(train_acc)
                # print(loss.item())
                self.epoch_loss[i, 0] = loss.item()

                # unlabeled loss
                unlabeled_loss = torch.mean((predicts - targets)**2)
                self.epoch_loss[i, 1] = unlabeled_loss.item()
                loss += unlabeled_loss * self.params['unsup_wght']

                # SNTG loss
                if self.params['embed']:
                    half = int(h_x.size()[0] // 2)
                    eucd2 = torch.mean((h_x[:half] - h_x[half:])**2, dim=1)
                    eucd = torch.sqrt(eucd2)
                    target_hard = torch.argmax(targets, dim=1).int()
                    merged_tar = torch.where(
                        mask == 0, target_hard, is_lens.int())
                    neighbor_bool = torch.eq(
                        merged_tar[:half], merged_tar[half:])
                    eucd_y = torch.where(eucd < 1.0, (1.0 - eucd) ** 2,
                                         torch.zeros_like(eucd))
                    embed_losses = torch.where(neighbor_bool, eucd2, eucd_y)
                    embed_loss = torch.mean(embed_losses)
                    self.epoch_loss[i, 2] = embed_loss.item()
                    loss += embed_loss * \
                        self.params['unsup_wght'] * self.params['embed_coeff']
                    self.epoch_loss[i, 3] = loss.item()
                loss.backward()
                self.optimizer.step()
                self.ema.update()

            self.ensemble_pred = \
                self.params['pred_decay'] * self.ensemble_pred + \
                (1 - self.params['pred_decay']) * self.epoch_pred
            self.targets_pred = self.ensemble_pred / \
                (1.0 - self.params['pred_decay'] ** (epoch + 1))
            loss_mean = torch.mean(self.epoch_loss, 0)
            train_losses.append(loss_mean[3].item())
            print(f"epoch {epoch}, time cosumed: {time.time() + train_time}, "
                  f"labeled loss: {loss_mean[0].item()}, "
                  f"unlabeled loss: {loss_mean[1].item()}, "
                  f"SNTG loss: {loss_mean[2].item()}, "
                  f"total loss: {loss_mean[3].item()}")

            # eval phase
            if self.eval_loader is not None:
                for i, data_batched in enumerate(self.eval_loader, epoch):
                    images, is_lens = \
                        data_batched['image'], data_batched['is_lens']
                    # currently h_x in evalization is not used
                    eval_logits, _ = self.ema(images)
                    test_acc = torch.mean(torch.argmax(
                        eval_logits, 1).eq(is_lens).float())
                    eval_accs.append(test_acc.item())
                    print(f"evaluation accuracy: {test_acc.item()}")
                    eval_lens = torch.zeros(
                        len(is_lens), is_lens.max()+1,
                        device=self.device) \
                        .scatter_(1, is_lens.unsqueeze(1), 1.)
                    # eval_loss = self.loss_fn(eval_logits, is_lens)
                    eval_loss = F.binary_cross_entropy_with_logits(eval_logits,
                                                                   eval_lens)
                    eval_losses.append(eval_loss.item())
                    break

        return train_losses, train_accs, eval_losses, eval_accs

    def test(self):
        self.net.eval()
        for i, data_batched in enumerate(self.test_loader, 0):
            images, is_lens = data_batched['image'], data_batched['is_lens']
            test_logits, _ = self.net(images)
            return roc_curve(is_lens, test_logits)
