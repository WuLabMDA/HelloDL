# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Classification")
    parser.add_argument("--device_id",        type=str,    default="0",
                        help="which gpu to use")
    parser.add_argument("--batch_size",       type=int,    default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test_batch_size",  type=int,    default=1000,
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs",           type=int,    default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr",               type=float,  default=1.0,
                        help="learning rate (default: 1.0)")
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
    
