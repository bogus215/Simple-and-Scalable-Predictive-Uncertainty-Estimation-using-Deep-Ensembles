# %% library
from easydict import EasyDict as edict
from model import *
from loader import notMNIST, loader
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

#%% data load
#not mnist data
args = edict()
args.path = 'D:/2020-2/비즈니스애널리틱스/논문리뷰/notMNIST_small'
args.batch_size = 32
args.loader= notMNIST(args)
X_abnormal , y_abnormal = iter(args.loader.train_iter).next()
#mnist data
args = edict()
args.batch_size = 32
args.seed = 0
args.loader= loader(args)
X_normal , y_normal = iter(args.loader.train_iter).next()

#%% model
# dropout
args = edict()
args.gpu_device = 0
model = Dropout(args).cuda(args.gpu_device)
model.load_state_dict(torch.load(os.path.join(
    'D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/parameter',
    'best_parameter_1206_dropout.pth')))
model.eval()
# ensemble
model1 = [Base(args).cuda(args.gpu_device) for _ in range(5)]
iter = 1
for model in model1:
    model.load_state_dict(torch.load(os.path.join(
        'D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/parameter',
        f'best_parameter_ensemble_{iter}.pth')))
    iter +=1
    model.eval()

# ensemble + AT
model2 = [Base(args).cuda(args.gpu_device) for _ in range(5)]
iter = 1
for model in model2:
    model.load_state_dict(torch.load(os.path.join(
        'D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/parameter',
        f'best_parameter_ensemble+AT_{iter}.pth')))
    iter +=1
    model.eval()

#%% entropy
# base model
X_normal = X_normal.cuda(args.gpu_device)
y_normal = y_normal.cuda(args.gpu_device)
X_abnormal = X_abnormal.cuda(args.gpu_device)
y_abnormal = y_abnormal.cuda(args.gpu_device)
normal_pred = F.softmax(model1[0](X_normal.view(X_normal.size(0),28*28)),dim=1)
normal_entropy = torch.sum(-normal_pred*torch.log2(normal_pred), axis=1).detach().cpu().numpy()
abnormal_pred = F.softmax(model1[0](X_abnormal.view(X_abnormal.size(0),28*28)),dim=1)
abnormal_entropy = torch.sum(-abnormal_pred*torch.log2(abnormal_pred), axis=1).detach().cpu().numpy()
sns.distplot(normal_entropy,hist=False).set_title('Base model entropy - normal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/Base_model_entropy_normal.png')
plt.show()
sns.distplot(abnormal_entropy,hist=False).set_title('Base model entropy - abnormal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/Base_model_entropy_abnormal.png')
plt.show()
# dropout
normal_pred = [F.softmax(model(X_normal.view(X_normal.size(0),28*28))
                         ,dim=1).view(1,X_normal.size(0),10) for _ in range(30)]
normal_pred = torch.cat(normal_pred).mean(axis=0)
normal_entropy = torch.sum(-normal_pred*torch.log2(normal_pred), axis=1).detach().cpu().numpy()
abnormal_pred = [F.softmax(model(X_abnormal.view(X_abnormal.size(0),28*28))
                         ,dim=1).view(1,X_abnormal.size(0),10) for _ in range(30)]
abnormal_pred = torch.cat(abnormal_pred).mean(axis=0)
abnormal_entropy = torch.sum(-abnormal_pred*torch.log2(abnormal_pred), axis=1).detach().cpu().numpy()
sns.distplot(normal_entropy,hist=False).set_title('dropout model entropy - normal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/dropout_model_entropy_normal.png')
plt.show()
sns.distplot(abnormal_entropy,hist=False).set_title('dropout model entropy - abnormal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/dropout_model_entropy_abnormal.png')
plt.show()
# ensemble
ensemble_normal_pred = torch.zeros((5,32,10))
ensemble_abnormal_pred = torch.zeros((5,32,10))
for ind, model in enumerate(model1):
    normal_pred = F.softmax(model(X_normal.view(X_normal.size(0), 28 * 28)), dim=1).cpu()
    ensemble_normal_pred[ind] = normal_pred
    abnormal_pred = F.softmax(model(X_abnormal.view(X_abnormal.size(0),28*28)),dim=1).cpu()
    ensemble_abnormal_pred[ind] = abnormal_pred
ensemble_normal_pred = ensemble_normal_pred.mean(axis=0)
ensemble_abnormal_pred = ensemble_abnormal_pred.mean(axis=0)
normal_entropy = torch.sum(-ensemble_normal_pred*torch.log2(ensemble_normal_pred), axis=1).detach().cpu().numpy()
abnormal_entropy = torch.sum(-ensemble_abnormal_pred*torch.log2(ensemble_abnormal_pred), axis=1).detach().cpu().numpy()
sns.distplot(normal_entropy,hist=False).set_title('ensemble model entropy - normal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/ensemble_model_entropy_normal.png')
plt.show()
sns.distplot(abnormal_entropy,hist=False).set_title('ensemble model entropy - abnormal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/ensemble_model_entropy_abnormal.png')
plt.show()
# ensemble + AT
ensemble_normal_pred = torch.zeros((5,32,10))
ensemble_abnormal_pred = torch.zeros((5,32,10))
for ind, model in enumerate(model2):
    normal_pred = F.softmax(model(X_normal.view(X_normal.size(0), 28 * 28)), dim=1).cpu()
    ensemble_normal_pred[ind] = normal_pred
    abnormal_pred = F.softmax(model(X_abnormal.view(X_abnormal.size(0),28*28)),dim=1).cpu()
    ensemble_abnormal_pred[ind] = abnormal_pred
ensemble_normal_pred = ensemble_normal_pred.mean(axis=0)
ensemble_abnormal_pred = ensemble_abnormal_pred.mean(axis=0)
normal_entropy = torch.sum(-ensemble_normal_pred*torch.log2(ensemble_normal_pred), axis=1).detach().cpu().numpy()
abnormal_entropy = torch.sum(-ensemble_abnormal_pred*torch.log2(ensemble_abnormal_pred), axis=1).detach().cpu().numpy()
sns.distplot(normal_entropy,hist=False).set_title('ensemble+AT model entropy - normal')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/ensemble+AT_model_entropy_normal.png')
plt.show()
sns.distplot(abnormal_entropy,hist=False).set_title('ensemble+AT model entropy - abnormal')
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/ensemble+AT_model_entropy_normal.png')
plt.xlim(-3,5)
plt.savefig('D:/2020-2/비즈니스애널리틱스/논문리뷰/Simple-and-Scalable-Predictive-Uncertainty-Estimation-using-Deep-Ensembles/img/ensemble+AT_model_entropy_abnormal.png')
plt.show()