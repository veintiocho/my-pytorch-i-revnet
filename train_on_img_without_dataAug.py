'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from base_models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
NET_NAME = 'SENet-image'
print('==> Building model..')
# net = VGG('VGG19',3)
# net = ResNet18(3)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
net = SENet18(3)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

def learning_rate(init, epoch, max_epoch):
    optim_factor = 0
    p_ratio = float(epoch)/float(max_epoch)
    if(p_ratio > 0.8):
        optim_factor = 3
    elif(p_ratio > 0.6):
        optim_factor = 2
    elif(p_ratio > 0.3):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)
# Training
def train(epoch, num_epochs):
    p_lr = learning_rate(args.lr,epoch,200)
    print('\nEpoch: %d Learning rate: %.4f' % (epoch, p_lr))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=p_lr, momentum=0.9, weight_decay=5e-4)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%% (%d/%d)' % (epoch, num_epochs, (train_loss/(batch_idx+1)), 100.*correct/total, correct, total))
        sys.stdout.flush()

def test(epoch):
    global best_acc, NET_NAME
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("\n| Testing...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sys.stdout.write('\r')
            sys.stdout.write('| Iter [%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%% (%d/%d)' % (batch_idx, len(testloader), (test_loss/(batch_idx+1)), 100.*correct/total, correct, total))
            sys.stdout.flush()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/' + NET_NAME):
            os.mkdir('checkpoint/' + NET_NAME)
        torch.save(state, 'checkpoint/' + NET_NAME + '/ckpt-' + str(int(acc)) + '.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch,200)
    test(epoch)