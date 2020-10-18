# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import optim
from torch.optim import lr_scheduler

from model import Net
from train_eng import net_train


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Classification")
    parser.add_argument("--device_id",        type=str,    default="0",
                        help="which gpu to use")
    parser.add_argument("--batch_size",       type=int,    default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test_batch_size",  type=int,    default=1000,
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--lr",               type=float,  default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--epochs",           type=int,    default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--log_interval",     type=int,    default=10,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--seed",             type=int,    default=1234,
                        help="random seed (default: 1234)")

    args = parser.parse_args()
    return args


def set_random_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Numpy
    np.random.seed(seed)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "...No available GPU..."
    args = parse_args()
    # set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    # ensure reproducible training
    set_random_seed(args.seed)

    # create model object
    net = Net()
    net = net.cuda()

    # prepare train & test dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_dset = datasets.MNIST("../data", train=False, transform=transform)
    train_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size, "num_workers": 4, "pin_memory": True, "shuffle": False}
    train_loader = torch.utils.data.DataLoader(train_dset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, **test_kwargs)

    # optimizer & scheduling
    optimizer = optim.SGD(net.parameters(), lr=arg.lr, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, gamma=0.7, step_size=1)

    # model training & test
    for epoch in range(1, args.epochs + 1):
        net_train(model, train_loader, optimizer, epoch, args)
        # net_test(model, test_loader)
        scheduler.step()
