# %% library
from loader import *
import argparse
from model import *
import numpy as np
import torch
import wandb
import torch.optim as optim
from rich.console import Console
import torch.nn.functional as F
from pytorchtools import EarlyStopping
from tqdm import tqdm
import gc
import random

# %% Train
def train(args):

    optimizer = optim.Adam(args.model.parameters(), args.learning_rate)
    best_nll = np.inf
    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.experiment}_{args.seed}.pth')
    steps_per_epoch = len(args.loader.train_iter)
    steps_per_epoch_test = len(args.loader.test_iter)
    console = Console()

    for e in range(1 , args.epoch + 1 ):
        print("\n===> epoch %d" % e)
        total_loss = 0
        with tqdm(total=steps_per_epoch, leave=False, dynamic_ncols=True) as pbar:
            for i, batch in enumerate(args.loader.train_iter):
                args.model.train()
                target = batch[1].cuda(args.gpu_device)
                feature = batch[0].cuda(args.gpu_device).view(target.size(0),28*28)
                optimizer.zero_grad()
                pred = F.softmax(args.model(feature),dim=1)
                loss = F.nll_loss(torch.log(pred),target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(1)

        if (e) % args.test == 0:
            with torch.no_grad():
                args.model.eval()
                nll=correct=total=bs=0
                with tqdm(total=steps_per_epoch_test, leave=False, dynamic_ncols=True) as pbar1:
                    for s, val_batch in enumerate(tqdm(args.loader.test_iter, desc='test')):
                        target = val_batch[1].cuda(args.gpu_device)
                        feature = val_batch[0].cuda(args.gpu_device).view(target.size(0), 28 * 28)
                        pred = F.softmax(args.model(feature), dim=1)
                        loss = F.nll_loss(torch.log(pred),target)
                        pbar1.update(1)
                        nll += loss.item()
                        _, predicted = torch.max(pred.data,1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        bs_temp = torch.zeros((target.size(0), 10)).cuda(args.gpu_device)
                        bs_temp[range(target.size(0)),target] = 1
                        bs += torch.mean(torch.square(pred-bs_temp),axis=[1,0])

            if best_nll > (nll / len(args.loader.test_iter)):
                best_nll = (nll / len(args.loader.test_iter))
                torch.save(args.model.state_dict(), f'./parameter/best_parameter_{args.experiment}_{args.seed}.pth')

            avg_loss = total_loss / len(args.loader.train_iter)
            avg_loss_val = nll / len(args.loader.test_iter)
            acc = (100 * correct) / total
            avg_bs = bs / len(args.loader.test_iter)
            console.print(f"Train [{e:>04}]/[{args.epoch:>03}]",f"Train NLL:{avg_loss:.4f}",end=' | ', style="Bold Cyan")
            console.print(f"Test NLL:{avg_loss_val:.4f}",f"Test ACC:{acc}",f"Test BS:{avg_bs}",sep=' | ', style='Bold Blue')
            wandb.log({'Train NLL': avg_loss,
                       'Test NLL': avg_loss_val,
                       'Test ACC': acc,
                       'Test BS' : avg_bs
                       })
            early_stopping(avg_loss_val, args.model)
            if early_stopping.early_stop:
                print('Early stopping')
                break

# %% main
def main():
    wandb.init(project='1206_Non_bayesian_uncertainty', reinit=True)
    parser = argparse.ArgumentParser(description="-----[#]-----")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epoch", default=300, type=int, help="number of max epoch")
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--test', default=2, type=int, help='test')
    parser.add_argument('--experiment', type=str, default='1206', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='seed')
    args = parser.parse_args()
    wandb.config.update(args)
    wandb.run.name = args.experiment +'_'+str(args.seed)
    wandb.run.save()
    args.loader = loader(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.model = Base(args).cuda(args.gpu_device)
    wandb.watch(args.model)
    gc.collect()
    train(args)
    wandb.finish()

# %% run
if __name__ == "__main__":
    main()