# %% library
from easydict import EasyDict as edict
from model import *
from loader import loader
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

#%% data load
#mnist data
args = edict()
args.batch_size = 32
args.seed = 0
args.gpu_device=0
args.loader= loader(args)

#%% ensemble
# ensemble
model1 = [Base(args).cuda(args.gpu_device) for _ in range(5)]
iter = 1
for model in model1:
    model.load_state_dict(torch.load(os.path.join(
        'D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/parameter',
        f'best_parameter_ensemble_{iter}.pth')))
    iter +=1
    model.eval()
nll = correct = total = bs = 0
for s, batch in enumerate(args.loader.test_iter):
    target = batch[1].cuda(args.gpu_device)
    feature = batch[0].cuda(args.gpu_device).view(target.size(0), 28 * 28)
    ensemble_normal_pred = torch.zeros((5,target.size(0),10)).cuda(args.gpu_device)
    for ind, model in enumerate(model1):
        normal_pred = F.softmax(model(feature.view(feature.size(0), 28 * 28)), dim=1)
        ensemble_normal_pred[ind] = normal_pred
    pred = ensemble_normal_pred.mean(axis=0)
    loss = F.nll_loss(torch.log(pred), target)
    nll += loss.item()
    _, predicted = torch.max(pred.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
    bs_temp = torch.zeros((target.size(0), 10)).cuda(args.gpu_device)
    bs_temp[range(target.size(0)), target] = 1
    bs += torch.mean(torch.square(pred - bs_temp), axis=[1, 0])
avg_loss_val = nll / len(args.loader.test_iter)
acc = (100 * correct) / total
avg_bs = bs / len(args.loader.test_iter)
print('negative log likelihood:',avg_loss_val)
print('Accuracy:',acc)
print('Brier score:',avg_bs)

# ensemble + AT

model2 = [Base(args).cuda(args.gpu_device) for _ in range(5)]
iter = 1
for model in model2:
    model.load_state_dict(torch.load(os.path.join(
        'D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/parameter',
        f'best_parameter_ensemble+AT_{iter}.pth')))
    iter +=1
    model.eval()
nll = correct = total = bs = 0
for s, batch in enumerate(args.loader.test_iter):
    target = batch[1].cuda(args.gpu_device)
    feature = batch[0].cuda(args.gpu_device).view(target.size(0), 28 * 28)
    ensemble_normal_pred = torch.zeros((5,target.size(0),10)).cuda(args.gpu_device)
    for ind, model in enumerate(model2):
        normal_pred = F.softmax(model(feature.view(feature.size(0), 28 * 28)), dim=1)
        ensemble_normal_pred[ind] = normal_pred
    pred = ensemble_normal_pred.mean(axis=0)
    loss = F.nll_loss(torch.log(pred), target)
    nll += loss.item()
    _, predicted = torch.max(pred.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
    bs_temp = torch.zeros((target.size(0), 10)).cuda(args.gpu_device)
    bs_temp[range(target.size(0)), target] = 1
    bs += torch.mean(torch.square(pred - bs_temp), axis=[1, 0])
avg_loss_val = nll / len(args.loader.test_iter)
acc = (100 * correct) / total
avg_bs = bs / len(args.loader.test_iter)
print('negative log likelihood:',avg_loss_val)
print('Accuracy:',acc)
print('Brier score:',avg_bs)