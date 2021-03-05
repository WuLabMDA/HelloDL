# -*- coding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn


def net_test(model, loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    error = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            test_loss += error(outputs, labels)
            _, predictions = outputs.max(1)
            correct += predictions.eq(labels).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    print("="*54)
    print("!!!Test Performance")
    print("Average loss: {:.6f}, Accuracy: {:-5d}/{} ({:.2f}%)".format(
        test_loss, correct, len(loader.dataset), accuracy))
    print("="*54)
