# -*- coding: utf-8 -*-

import os, sys
import torch.nn.functional as F


def net_train(model, loader, optimizer, epoch, args):
    model.train()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
