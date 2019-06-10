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


def extract_feat(model, trainloader, testloader, save_name):
    model.eval()
    number = 0
    for idx, (input, target) in enumerate(trainloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        _, _, output_32x32_0 = model(input_var)

        output_32x32_0_cpu = output_32x32_0.cpu().detach().numpy()

        target_cpu = target.cpu().detach().numpy()

        batch_size = output_32x32_0_cpu.shape[0]
        # print(batch_size)
        for i in range(batch_size):
            feat0 = output_32x32_0_cpu[i,:,:,:]
            
            label = target_cpu[i]

            save_folder_0 = 'data/' + save_name
            if(os.path.exists(save_folder_0)==False):
                os.mkdir(save_folder_0)

            save_path_0 = save_folder_0 + '/' + str(label) + '/'    
            if(os.path.exists(save_path_0)==False):
                os.mkdir(save_path_0)

            np.save(save_path_0 + str(number) + '.npy',feat0)

            number+=1
        
        sys.stdout.write('\r')
        sys.stdout.write('| Iter %3d' % (idx+1))
        sys.stdout.flush()


    number = 0
    for idx, (input, target) in enumerate(testloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        _, _, output_32x32_0 = model(input_var)

        output_32x32_0_cpu = output_32x32_0.cpu().detach().numpy()

        target_cpu = target.cpu().detach().numpy()

        batch_size = output_32x32_0_cpu.shape[0]
        # print(batch_size)
        for i in range(batch_size):
            feat0 = output_32x32_0_cpu[i,:,:,:]
            
            label = target_cpu[i]

            save_folder_0 = 'data/' + save_name
            if(os.path.exists(save_folder_0)==False):
                os.mkdir(save_folder_0)

            save_path_0 = save_folder_0 + '/' + str(label) + '/'
      
            if(os.path.exists(save_path_0)==False):
                os.mkdir(save_path_0)

            np.save(save_path_0 + str(number) + '.npy',feat0)

            number+=1
        sys.stdout.write('\r')
        sys.stdout.write('| Iter %3d' % (idx+1))
        sys.stdout.flush()


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', default="vgg", type=str, help='networks')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

print('==> Building model..')
if (args.net == 'vgg'):
    NET_NAME = 'VGG19-image'
    net = VGG('VGG19',3)
elif (args.net == 'resnet'):
    NET_NAME = 'VGG19-image'
    net = VGG('VGG19',3)
else:
    "Not supported network!!"
    exit(0)
# Model
# net = VGG('VGG19')
# net = ResNet18(3)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if os.path.isfile(args.resume):
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print("=> no checkpoint found at '{}'".format(args.resume))


extract_feat(net,trainloader,testloader,NET_NAME)