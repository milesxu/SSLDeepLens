import math
import torch.optim as opt


def learning_rate_update(rampup_length, rampdown_length, learning_rate,
                         adam_beta1, rd_beta1_target, num_epochs, scale=0.01):
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
        unsup_weight = 0.0 if epoch == 0 else ru * scale
        return unsup_weight

    return adam_param_update
