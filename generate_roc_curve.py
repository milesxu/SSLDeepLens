import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import ground_based_dataset as gbd
import resnet_ssl_model as rsm
from data_transforms import Log10, Clamp, AugmentTranslate, WhitenInput
from run_loop import SNTGRunLoop


def now_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


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
# print(prob.shape)
# print(prob[0], prob[1])
prob_flat = prob.flatten()
# print(prob_flat.shape)
# print(prob_flat[0], prob_flat[1])
stride = np.linspace(0, 1, 1000)
p = np.where(prob > stride, 1, 0)
# print(labels.size())
# print(labels[0], labels[1])
labels = labels.cpu().numpy().astype('bool')
# print(labels[0], labels[1])
# labels = np.array(labels, dtype=bool)
# print(labels.shape)
# print(len(labels))
fpr_sk, tpr_sk, _ = roc_curve(labels, prob_flat)
# print(fpr_sk)
# print(tpr_sk)
print(auc(fpr_sk, tpr_sk))
# print(now_str())
plt.figure()
lw = 1
plt.plot(fpr_sk, tpr_sk, color='darkgreen', lw=lw,
         label=f"ROC curve (area = {auc(fpr_sk, tpr_sk)})")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SNTG ResNet50 ROC for ground based data')
plt.legend(loc='lower right')

# n_fp = np.sum(p * (labels == 0), axis=0).astype('float32')
# n_tp = np.sum(p * labels, axis=0).astype('float32')
# fp_array = p[~labels, :]
# tp_array = p[labels, :]
# print(len(fp_array), len(tp_array))
# print(fp_array)
# print(tp_array)
# n_fp = np.sum(fp_array, axis=0).astype('float32')
# n_tp = np.sum(tp_array, axis=0).astype('float32')

# labels = np.array(labels, dtype='int')
# print(labels)
# tpr = n_tp / np.sum(labels).astype('float32')
# fpr = n_fp / np.sum(labels == 0).astype('float32')

# plt.suptitle('ROC on Training set')
# plt.plot(fpr, tpr)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid('on')
img_name = 'roc_curve_' + now_str() + '.png'
plt.savefig(os.path.join(data_path, img_name), format='png')
