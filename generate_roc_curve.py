import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import ground_based_dataset as gbd
import resnet_ssl_model as rsm
from data_transforms import Log10, Clamp, AugmentTranslate, WhitenInput
from run_loop import SNTGRunLoop

torch.manual_seed(770715)
torch.cuda.manual_seed_all(770715)
data_path = '/data/mingx/datasets'
test_composed = transforms.Compose([WhitenInput(), Clamp(1e-9, 100)])

sntg_net = rsm.SNTGModel(4)
sntg_model = 'saved_model/ground_based2019-05-17-12-02.pth'
sntg_net.load_state_dict(torch.load(sntg_model))

start, length = 12000, 2048
dataset = gbd.GroundBasedDataset(data_path, offset=start, length=length,
                                 transform=test_composed)
data_loader = DataLoader(dataset, batch_size=length,
                         shuffle=False, pin_memory=False)
run_loop = SNTGRunLoop(sntg_net, test_loader=data_loader)
logits, labels = run_loop.test_origin()
logits = F.softmax(logits, dim=1)
# print(logits.sum(1))
prob = logits[:, 1].cpu().numpy()
prob = np.reshape(prob, (-1, 1))
stride = np.linspace(0, 1, 1000)
p = np.where(prob > stride, 1, 0)
# print(labels.size())
labels = labels.cpu().numpy().astype('bool')
# labels = np.array(labels, dtype=bool)
print(len(labels))

# n_fp = np.sum(p * (labels == 0), axis=0).astype('float32')
# n_tp = np.sum(p * labels, axis=0).astype('float32')
fp_array = p[~labels, :]
tp_array = p[labels, :]
print(len(fp_array), len(tp_array))
print(fp_array)
print(tp_array)
n_fp = np.sum(fp_array, axis=0).astype('float32')
n_tp = np.sum(tp_array, axis=0).astype('float32')

labels = np.array(labels, dtype='int')
print(labels)
tpr = n_tp / np.sum(labels).astype('float32')
fpr = n_fp / np.sum(labels == 0).astype('float32')

plt.suptitle('ROC on Training set')
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid('on')
plt.savefig(os.path.join(data_path, 'roc_curve.png'), format='png')
