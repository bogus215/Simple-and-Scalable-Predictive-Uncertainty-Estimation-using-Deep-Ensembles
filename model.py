#%% lib
import torch.nn as nn
import torch.nn.functional as F
'''
 For MNIST, we used an MLP with 3-hidden layers with 200 hidden units per layer and ReLU non-linearities with batch
normalization. For MC-dropout, we added dropout after each non-linearity with 0.1 as the dropout
rate.7 Results are shown in Figure 2(a). We observe that adversarial training and increasing the
number of networks in the ensemble significantly improve performance in terms of both classification
accuracy as well as NLL and Brier score, illustrating that our method produces well-calibrated
uncertainty estimates.
'''

#%% base
class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.input = nn.Linear(28*28,200)
        self.batch0 = nn.BatchNorm1d(200)
        self.hidden1 = nn.Linear(200,200)
        self.batch1 = nn.BatchNorm1d(200)
        self.hidden2 = nn.Linear(200,200)
        self.batch2 = nn.BatchNorm1d(200)
        self.hidden3 = nn.Linear(200,200)
        self.batch3 = nn.BatchNorm1d(200)
        self.final = nn.Linear(200,10)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.input(x)
        x = self.relu(self.batch0(x))
        x = self.hidden1(x)
        x = self.relu(self.batch1(x))
        x = self.hidden2(x)
        x = self.relu(self.batch2(x))
        x = self.hidden3(x)
        x = self.relu(self.batch3(x))
        x = self.final(x)
        return x

#%% Dropout
class Dropout(nn.Module):
    def __init__(self, args):
        super(Dropout, self).__init__()

        self.input = nn.Linear(28*28,200)
        self.batch0 = nn.BatchNorm1d(200)
        self.hidden1 = nn.Linear(200,200)
        self.batch1 = nn.BatchNorm1d(200)
        self.hidden2 = nn.Linear(200,200)
        self.batch2 = nn.BatchNorm1d(200)
        self.hidden3 = nn.Linear(200,200)
        self.batch3 = nn.BatchNorm1d(200)
        self.final = nn.Linear(200,10)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = MCDropout(p = 0.1 , force_dropout=True)

    def forward(self,x):
        x = self.input(x)
        x = self.relu(self.batch0(x))
        x = self.dropout(x)
        x = self.hidden1(x)
        x = self.relu(self.batch1(x))
        x = self.dropout(x)
        x = self.hidden2(x)
        x = self.relu(self.batch2(x))
        x = self.dropout(x)
        x = self.hidden3(x)
        x = self.relu(self.batch3(x))
        x = self.dropout(x)
        x = self.final(x)
        return x



class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5, force_dropout: bool = False):
        super().__init__()
        self.force_dropout = force_dropout
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training or self.force_dropout)
        # return nn.functional.dropout2d(x, p=self.p, training=self.training or self.force_dropout)
