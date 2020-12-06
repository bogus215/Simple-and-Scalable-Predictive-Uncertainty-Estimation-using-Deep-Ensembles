import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader , random_split
import torch
import numpy as np
import random
import torchvision.datasets as datasets


class loader():

    def __init__(self, args):
        super(loader, self).__init__()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mnist_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=0.5,std=0.5)])
        download_root = 'D:/2020-2/비즈니스애널리틱스/논문리뷰/MNIST_DATASET'

        dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
        train_dataset , valid_dataset = random_split(dataset , [50000,10000])
        test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=self.batch_size , shuffle=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=self.batch_size , shuffle=True)
        del train_dataset, valid_dataset, test_dataset

class notMNIST():
    def __init__(self,args):

        download_root = args.path
        dataset = datasets.ImageFolder(
            download_root,
            transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)]))


        train_dataset , valid_dataset = random_split(dataset , [int(len(dataset)*0.5),len(dataset) - int(len(dataset)*0.5)])
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=args.batch_size , shuffle=True , pin_memory=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=args.batch_size , shuffle=True , pin_memory=True)
