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


def eval_outputs(p_in, labels_in):
    '''
    calculate FPR and TPR according to the 
    output probability and original labels.
        - <p_in> is of the outputs of the probability 
          of the images.
        - <labels_in> is of the original labels of 
          the input images. 
    '''
    p_index_sorted = np.argsort(p_in)
    p_in_sorted = p_in[p_index_sorted]
    labels_in_sorted = labels_in[p_index_sorted]
    fpr_out = p_in*0.0
    tpr_out = p_in*0.0
    nNonLenses = len(labels_in[np.where(labels_in == 0)])
    nLenses = len(labels_in[np.where(labels_in == 1)])
    for i in range(len(p_in)):
        masksNonLenses = labels_in_sorted[i:] == 0
        masksLenses = labels_in_sorted[i:] == 1
        fpr_out[i] = len(labels_in_sorted[i:][masksNonLenses])/nNonLenses
        tpr_out[i] = len(labels_in_sorted[i:][masksLenses])/nLenses
    return fpr_out, tpr_out


def tprN(nFP, p_in, labels_in):
    '''
    Calculate TPR0, TPR10, and beyond. 
        - <nFP> is the threshold of False positives, 
          e.g., for tpr0, nFP = 1, 
          for tpr10, nFP = 10.
        - <p_in> is the outputs of the probability 
          of each image.
        - <labels_in> is of the original labels of 
          the input images. 
    '''
    p_index_sorted = np.argsort(p_in)
    p_in_sorted = p_in[p_index_sorted]
    labels_in_sorted = labels_in[p_index_sorted]
    nLenses = len(labels_in[np.where(labels_in == 1)])
    for i in range(1, len(p_in)):
        idx = labels_in_sorted[-i:] < 1
        nFPs = len(labels_in_sorted[-i:][idx])
        if nFPs >= nFP:
            break
        else:
            p_th = p_in_sorted[-i]
            nTPs = len(labels_in_sorted[-i:][~idx])
            tprN = nTPs/nLenses
    return tprN


def save_npz(labels, stride, prob_flat, save=False, npz_path=None):
    fp_array = p[~labels, :]
    tp_array = p[labels, :]
    n_fp = np.sum(fp_array, axis=0).astype('float32')
    n_tp = np.sum(tp_array, axis=0).astype('float32')

    labels = np.array(labels, dtype='int')
    tpr = n_tp / np.sum(labels).astype('float32')
    fpr = n_fp / np.sum(labels == 0).astype('float32')
    if save:
        npz_path = os.path.join(data_path, 'tpr_result.npz')
        np.savez(npz_path, tpr=tpr, fpr=fpr, stride=stride,
                 prob=prob_flat, labels=labels)
    return tpr, fpr


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
prob = logits[:, 1].cpu().numpy()
prob = np.reshape(prob, (-1, 1))
prob_flat = prob.flatten()
stride = np.linspace(0, 1, 1000)
p = np.where(prob > stride, 1, 0)
labels = labels.cpu().numpy().astype('bool')
# labels = np.array(labels, dtype=bool)
tpr, fpr = save_npz(labels, stride, prob_flat)
fpr_sk, tpr_sk, _ = roc_curve(labels, prob_flat)
print(auc(fpr_sk, tpr_sk))
fpr_arr, tpr_arr = eval_outputs(prob_flat, labels)
masks1 = fpr < 1.0
masks2 = fpr_arr < 1.0
masks3 = fpr_sk < 1.0
lw = 1
plt.figure()
# plt.plot(fpr_sk, tpr_sk, color='darkgreen', lw=lw,
#  label=f"ROC curve (area = {auc(fpr_sk, tpr_sk)})")
plt.plot(fpr[masks1], tpr[masks1], '.', label='Your Results')
plt.plot(fpr_arr[masks2], tpr_arr[masks2], '-', label='this function')
plt.plot(fpr_sk[masks3], tpr_sk[masks3], '-', label='sklearn')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SNTG ResNet50 ROC for ground based data')
plt.legend(loc='lower right')


# plt.suptitle('ROC on Training set')
# plt.plot(fpr, tpr)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid('on')
img_name = 'roc_curve_' + now_str() + '.png'
plt.savefig(os.path.join(data_path, img_name), format='png')

# -------------------------------------------------------
# Calculating tpr0 and tpr10.
# Note: the definitions of tpr0 and tpr10 are for the
# testing set of 100k images only, hence,
# the tpr0 and tpr10 for the dataset you
# provided is not comparable to the numbers
# in the paper of lens Finding Challenge.
# If we want to evaluate the rank of our
# approach in lens finding challenge, we
# have to apply this function to the outputs
# of the testing set of 100k images from
# lens finding challenge.
tpr0 = tprN(1, prob_flat, labels)
tpr10 = tprN(10, prob, labels)
print(tpr0, tpr10)
