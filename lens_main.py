import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import ground_based_dataset as gbd
import resnet_ssl_model as rsm
from data_transforms import Log10, Clamp, AugmentTranslate, WhitenInput
from run_loop import SNTGRunLoop
from learning_rate_update import learning_rate_update

path = ''
if sys.platform == 'linux':
    path = '/home/mingx/datasets/'
else:
    path = 'C:\\Users\\miles\\Documents\\dataset'
save_path = os.path.join(path, 'saved_model')
train_params = {
    'n_data': 5120,
    'num_classes': 2,
    'batch_size': 128,
    'n_eval_data': 1024,
    'test_offset': 10000,
    'test_len': 1000,
    'run_eval': True,
    'run_test': False,
    'num_epochs': 100,
    'rampup_length': 80,
    'rampdown_length': 50,
    'learning_rate': 0.003,
    'pred_decay': 0.6,
    'embed': True,
    'embed_coeff': 0.2,
    'adam_beta1': 0.9,
    'rd_beta1_target': 0.5,
    'augment_mirror': True,
    'augment_translation': 2,
    'unsup_wght': 0.0,
    'whiten_inputs': 'norm',  # norm, zca
    'polyak_decay': 0.999
}
torch.manual_seed(770715)
has_cuda = torch.cuda.is_available()
if has_cuda:
    torch.cuda.manual_seed_all(770715)
    torch.backends.cudnn.deterministic = True
train_composed = transforms.Compose(
    [WhitenInput(), Clamp(1e-9, 100),
     AugmentTranslate(train_params['augment_translation'], 101)])
test_composed = transforms.Compose([WhitenInput(), Clamp(1e-9, 100)])
ground_train_dataset = gbd.GroundBasedDataset(
    path, length=train_params['n_data'], transform=train_composed)
ground_train_loader = DataLoader(
    ground_train_dataset, batch_size=train_params['batch_size'],
    shuffle=True, pin_memory=not has_cuda)
ground_eval_dataset = gbd.GroundBasedDataset(
    path, offset=train_params['n_data'], length=train_params['n_eval_data'],
    transform=test_composed)
ground_eval_loader = DataLoader(
    ground_eval_dataset, batch_size=train_params['batch_size'],
    shuffle=False, pin_memory=not has_cuda)
ground_test_dataset = gbd.GroundBasedDataset(
    path, offset=train_params['test_offset'], length=train_params['test_len'],
    transform=test_composed)
ground_test_loader = DataLoader(
    ground_test_dataset, batch_size=train_params['test_len'],
    pin_memory=not has_cuda)


# ssl_lens_net = rsm.ResNetSSL([3, 3, 3, 3, 3])
ssl_lens_net = rsm.SNTGModel(4)

# labeled_loss = nn.BCEWithLogitsLoss()

lr_fn = learning_rate_update(
    train_params['rampup_length'], train_params['rampdown_length'],
    train_params['learning_rate'], train_params['adam_beta1'],
    train_params['rd_beta1_target'], train_params['num_epochs']
    # train_params['rd_beta1_target'], 100
)

rnssl_run_loop = SNTGRunLoop(
    ssl_lens_net, ground_train_loader, train_params, lr_fn,
    eval_loader=ground_eval_loader, test_loader=ground_test_loader,
    has_cuda=has_cuda)

train_losses, train_accs, eval_losses, eval_accs, \
    ema_eval_losses, ema_eval_accs = rnssl_run_loop.train()
# rnssl_run_loop.test()
loss_df = pd.DataFrame(
    data={'epoch': np.arange(train_params['num_epochs']),
          'train loss': train_losses, 'evaluation loss': eval_losses,
          'ema evaluation loss': ema_eval_losses})
acc_df = pd.DataFrame(
    data={'epoch': np.arange(train_params['num_epochs']),
          'train accuracy': train_accs, 'evaluation accuracy': eval_accs,
          'ema evaluation accuracy': ema_eval_accs})

if not os.path.isdir(save_path):
    os.mkdir(save_path)
file_name = 'ground_based' + \
    datetime.datetime.now().isoformat(
        '-', timespec='minutes').replace(':', '-') + '.pth'
torch.save(ssl_lens_net.state_dict(), os.path.join(save_path, file_name))

# test_net = rsm.ResNetSSL([3, 3, 3, 3, 3])
# test_net.load_state_dict(torch.load(os.path.join(save_path, file_name)))
# test_net.eval()
