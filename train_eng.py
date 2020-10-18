# -*- coding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def net_train(model, loader, optimizer, epoch, args):
    model.train()
    error = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            cur_nums = batch_idx * len(images)
            cur_per = int(100. * batch_idx / len(loader))
            print("Train Epoch: {:-2d} [{:-5d}/{} ({:-3d}%)]\tLoss: {:.6f}".format(
                epoch, cur_nums, len(loader.dataset), cur_per, loss.item() / len(labels)))
