'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import argparse
import logging
import time
import logging
import sys

from models.resnet import ResNet18
from models.cnn import CNN

from RA.augmentations import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--exp', default='debug', type=str, help='experiment name')
parser.add_argument('--backbone', default='R18', type=str, help='backbone architecture')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=300, type=int, help='epochs number')
parser.add_argument('--riesz-gamma', default=1e-4, type=float, help='riesz loss weight')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ada-alpha', '-aa', action='store_true', help='apply adaptive alpha in riesz loss')
parser.add_argument('--rand-aug', '-ra', action='store_true', help='apply RandAugment')
args = parser.parse_args()

exp_path = '../exps/'+args.exp
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

checkpoints_path = os.path.join(exp_path, 'checkpoint')
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler(os.path.join(exp_path, 'logs.txt'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.rand_aug:
    transform_train.transforms.insert(0, RandAugment(n=3, m=5))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
print('==> {} backbone.'.format(args.backbone))
if args.backbone == 'R18':
    net = ResNet18(args.ada_alpha)
elif args.backbone[:3] == 'CNN':
    num_layers = int(args.backbone[3:])
    net = CNN(num_layers, args.ada_alpha)
else:
    raise Exception('Unknown backbone architecture!')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoints_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoints_path, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225, 270], gamma=0.1)


# Training
def train(epoch, gamma):
    #if epoch < 10: gamma = 0

    logger.info('Epoch: %d' % epoch)
    net.train()
    train_ce_loss = 0
    train_riesz_loss = 0
    train_energy = 0
    train_alpha = 0
    correct = 0
    total = 0
    tic = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, riesz_losses, energies, alphas = net(inputs)
        ce_loss = criterion(outputs, targets)
        riesz_loss = torch.stack(riesz_losses).mean()
        loss = ce_loss + gamma * riesz_loss
        loss.backward()
        optimizer.step()

        train_ce_loss += ce_loss.item()
        train_riesz_loss += riesz_loss.item()
        train_energy += torch.stack(energies).mean()
        train_alpha += torch.stack(alphas).mean()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    toc = time.time()
    logger.info('TRAIN | CE_Loss: %.3f | Riesz_Loss: %.3f | Acc: %.3f%% (%d/%d) | Energy: %.3f | Alpha: %.2f | Time: %d s' % \
            (train_ce_loss/(batch_idx+1), train_riesz_loss/(batch_idx+1), 100.*correct/total, correct, total,\
            train_energy/(batch_idx+1), train_alpha/(batch_idx+1), toc-tic))

def test(epoch):
    global best_acc
    net.eval()
    test_ce_loss = 0
    test_riesz_loss = 0
    correct = 0
    total = 0
    tic = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, riesz_losses, _, _ = net(inputs)
            ce_loss = criterion(outputs, targets)
            riesz_loss = torch.stack(riesz_losses).mean()

            test_ce_loss += ce_loss.item()
            test_riesz_loss += riesz_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    toc = time.time()
    logger.info('TEST | CE_Loss: %.3f | Riesz_Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %d s' % \
               (test_ce_loss/(batch_idx+1), test_riesz_loss/(batch_idx+1), 100.*correct/total, correct, total, toc-tic))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoints_path, 'ckpt.pth'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch, gamma=args.riesz_gamma)
    test(epoch)
    scheduler.step()
