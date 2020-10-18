# -*- coding: utf-8 -*-

import os, sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Classification")
    parser.add_argument("--batch-size",       type=int,    default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size",  type=int,    default=1000,
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs",           type=int,    default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument('--lr',               type=float,  default=1.0,
                        help="learning rate (default: 1.0)")

    args = parser.parse_args()

    return args
