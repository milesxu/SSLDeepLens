import datetime
import time
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F


class TrainLoop(object):
    def __init__(self, net, dataloader, params, device, update_fn):
        self.net = net
        self.loader = dataloader
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
        self.epoch_loss = t_one.new_empty(
            (n_data // params['batch_size'], 4), dtype=torch.float32, device=device)
        self.optimizer = opt.Adam(self.net.parameters())
        self.update_fn = update_fn

    def __call__(self, is_training=True):
        if is_training:
            self.net.train()
        else:
            self.net.eval()
        labeled_loss = nn.CrossEntropyLoss()
        for epoch in range(self.params['num_epochs']):
            train_time = -time.time()
            self.epoch_pred.zero_()
            self.epoch_mask.zero_()
            # epoch_loss.zero_()
            if is_training:
                self.update_fn(self.optimizer, epoch)

            for i, data_batched in enumerate(self.loader, 0):
                images, is_lens, mask, indices = \
                    data_batched['image'], data_batched['is_lens'], \
                    data_batched['mask'], data_batched['index']

                targets = torch.index_select(self.target_pred, 0, indices)

                self.optimizer.zero_grad()

                outputs, h_x = self.net(images)
                #print(torch.mean(h_x), torch.mean(outputs))
                # print(torch.max(h_x), torch.max(outputs))
                # print(torch.min(h_x), torch.min(outputs))
                predicts = F.softmax(outputs, dim=1)

                # update for ensemble
                for k, j in enumerate(indices):
                    self.epoch_pred[j] = predicts[k]
                    self.epoch_mask[j] = 1.0

                # labeled loss
                labeled_mask = mask.eq(0)
                #print(f'labeled: {labeled_mask} and unlabeled: {mask}')
                loss = labeled_loss(
                    outputs[labeled_mask], is_lens[labeled_mask])
                # print(loss.item())
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
                    merged_tar = torch.where(
                        mask == 0, target_hard, is_lens.int())
                    neighbor_bool = torch.eq(
                        merged_tar[:half], merged_tar[half:])
                    eucd_y = torch.where(eucd < 1.0, (1.0 - eucd) ** 2,
                                         torch.zeros_like(eucd))
                    embed_losses = torch.where(neighbor_bool, eucd2, eucd_y)
                    embed_loss = torch.mean(embed_losses)
                    epoch_loss[i, 2] = embed_loss.item()
                    loss += embed_loss * unsup_wght * embed_coeff
                    epoch_loss[i, 3] = loss.item()
                loss.backward()
                optimizer.step()
                ema.update()

                # print(loss, unlabeled_loss)

            ensemble_pred = pred_decay * ensemble_pred + \
                (1 - pred_decay) * epoch_pred
            targets_pred = ensemble_pred / (1.0 - pred_decay ** (epoch + 1))
            # print(epoch_loss)
            loss_mean = torch.mean(epoch_loss, 0)
            print(f"epoch {epoch}, time cosumed: {time.time() + train_time}, "
                  f"labeled loss: {loss_mean[0].item()}, "
                  f"unlabeled loss: {loss_mean[1].item()}, "
                  f"SNTG loss: {loss_mean[2].item()}, "
                  f"total loss: {loss_mean[3].item()}")

            if run_eval:
                #eval_images, eval_is_lens, _, _ = eval_iter.next()
                #print(eval_images.size(), eval_is_lens.size())
                for i, data_batched in enumerate(ground_eval_loader, epoch):
                    images, is_lens = data_batched['image'], data_batched['is_lens']
                    eval_logits, eval_h_embed = ema(images)
                    test_acc = torch.mean(torch.argmax(
                        eval_logits, 1).eq(is_lens).float())
                    print(f"evaluation accuracy: {test_acc.item()}")
                    break
