import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
np.random.seed(0)
import os
import argparse
import logging
import time
import logging
import sys

from models.resnet import ResNet18, ResNet34, ResNet50
from models.cnn import CNN
from regularization.baselines import *
from regularization.frames import *

#from RA.augmentations import *
from utils import load_config


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('config', type=str, help='experiment config')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()

cfg = load_config(args.config)

exp_path = '../exps/'+cfg.EXP
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

device = 'cuda:{}'.format(args.gpu)
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

#if cfg.RA:
#    transform_train.transforms.insert(0, RandAugment(n=3, m=5))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if cfg.TRAIN.DATASET == 'cifar10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
elif cfg.TRAIN.DATASET == 'cifar100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test)

subset_indices = np.arange(len(trainset))
np.random.shuffle(subset_indices)
subset_size = int(len(trainset) * cfg.TRAIN.SUBSET_RATIO)
trainset = torch.utils.data.Subset(trainset, subset_indices[:subset_size])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
print('==> {} backbone.'.format(cfg.TRAIN.BACKBONE))
if cfg.TRAIN.BACKBONE == 'R18':
    net = ResNet18(num_classes=num_classes)
elif cfg.TRAIN.BACKBONE == 'R34':
    net = ResNet34(num_classes=num_classes)
elif cfg.TRAIN.BACKBONE[:3] == 'CNN':
    num_layers = int(cfg.TRAIN.BACKBONE[3:])
    assert num_layers % 3 == 0
    net = CNN(num_layers//3, num_classes=num_classes)
else:
    raise Exception('Unknown backbone architecture!')

net = net.to(device)
if 'cuda' in device:
    net = torch.nn.DataParallel(net, device_ids=[args.gpu])
    cudnn.benchmark = True

if cfg.WEIGHTS_REGULARIZATION.MHE.ENABLED:
    weights_loss_name = 'MHE'
    weights_reguralizer = hyperspherical_energy
elif cfg.WEIGHTS_REGULARIZATION.SO.ENABLED:
    weights_loss_name = 'SO'
    weights_reguralizer = SO
elif cfg.WEIGHTS_REGULARIZATION.SRIP.ENABLED:
    weights_loss_name = 'SRIP'
    weights_reguralizer = SRIP
elif cfg.WEIGHTS_REGULARIZATION.DDRL.ENABLED:
    weights_loss_name = 'DDRL'
    weights_reguralizer = lambda x: diag_dominance_RL(x, cfg.WEIGHTS_REGULARIZATION.DDRL.ALPHA, cfg.WEIGHTS_REGULARIZATION.DDRL.BETA)
elif cfg.WEIGHTS_REGULARIZATION.EIGRL.ENABLED:
    weights_loss_name = 'EIGRL'
    weights_reguralizer = lambda x: eigs_RL(x, cfg.WEIGHTS_REGULARIZATION.EIGRL.ALPHA, cfg.WEIGHTS_REGULARIZATION.EIGRL.BETA)
elif cfg.WEIGHTS_REGULARIZATION.TOEPDDRL.ENABLED:
    weights_loss_name = 'TOEPDDRL'
    weights_reguralizer = lambda x: eigs_RL(x, cfg.WEIGHTS_REGULARIZATION.TOEPDDRL.ALPHA, cfg.WEIGHTS_REGULARIZATION.TOEPDDRL.BETA)
else:
    weights_reguralizer = None

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=5e-4)
milestones = (cfg.TRAIN.EPOCHS * np.array([0.5, 0.75, 0.9])).astype(int)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


def reguralizer_scale_modifier(epoch, gamma, mode='shortened'):
    stage_gammas = [gamma, 1e-5, 1e-6, 1e-8, 0]
    if mode == 'shortened':
        assert gamma == 0.1
        milestones = (cfg.TRAIN.EPOCHS * np.array([0.05, 0.075, 0.15, 0.25])).astype(int)
    elif mode == 'original':
        assert gamma == 0.01
        milestones = (cfg.TRAIN.EPOCHS * np.array([0.1, 0.25, 0.35, 0.6])).astype(int)
    else:
        raise Exception('Unknown reguralizer schedule mode!')

    stage = np.sum(milestones > epoch)
    gamma = stage_gammas[stage]
    return gamma


def train(epoch, gamma):
    if cfg.WEIGHTS_REGULARIZATION.SCHEDULER.ENABLED:
        gamma = reguralizer_scale_modifier(epoch, gamma, cfg.WEIGHTS_REGULARIZATION.MODE)

    logger.info('Epoch: %d' % epoch)
    net.train()
    train_ce_loss = 0
    train_weights_loss = 0
    correct = 0
    total = 0
    tic = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        ce_loss = criterion(outputs, targets)
        if weights_reguralizer is not None:
            weights_losses = [weights_reguralizer(w) for w in net.parameters() if len(w.shape)==4]
            weights_loss = torch.stack(weights_losses).mean()
            loss = ce_loss + gamma * weights_loss
        else:
            loss = ce_loss
        loss.backward()
        optimizer.step()

        train_ce_loss += ce_loss.item()
        if weights_reguralizer is not None:
            train_weights_loss += weights_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    toc = time.time()
    if weights_reguralizer is not None:
        info = 'TRAIN | CE_Loss: %.3f | %s: %.3f | Acc: %.3f%% (%d/%d) | Time: %d s' % \
                (train_ce_loss/(batch_idx+1), weights_loss_name, train_weights_loss/(batch_idx+1), 100.*correct/total, correct, total, toc-tic)
    else:
        info = 'TRAIN | CE_Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %d s' % \
                (train_ce_loss/(batch_idx+1),  100.*correct/total, correct, total, toc-tic)

    logger.info(info)

def test(epoch):
    global best_acc
    net.eval()
    test_ce_loss = 0
    test_weights_loss = 0
    correct = 0
    total = 0
    tic = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            ce_loss = criterion(outputs, targets)
            test_ce_loss += ce_loss.item()

            if weights_reguralizer is not None:
                weights_losses = [weights_reguralizer(w) for w in net.parameters() if len(w.shape)==4]
                weights_loss = torch.stack(weights_losses).mean()
                test_weights_loss += weights_loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    toc = time.time()

    if weights_reguralizer is not None:
        info = 'TEST | CE_Loss: %.3f | %s: %.3f | Acc: %.3f%% (%d/%d) | Time: %d s' % \
               (test_ce_loss/(batch_idx+1), weights_loss_name, test_weights_loss/(batch_idx+1), 100.*correct/total, correct, total, toc-tic)
    else:
        info = 'TEST | CE_Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %d s' % \
               (test_ce_loss/(batch_idx+1), 100.*correct/total, correct, total, toc-tic)

    logger.info(info)

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


for epoch in range(cfg.TRAIN.EPOCHS):
    train(epoch, gamma=cfg.WEIGHTS_REGULARIZATION.GAMMA)
    test(epoch)
    scheduler.step()
