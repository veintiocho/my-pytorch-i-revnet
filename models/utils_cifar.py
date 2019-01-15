# Author: J.-H. Jacobsen.
#
# Modified from Pytorch examples code.
# Original license shown below.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from models.model_utils import get_all_params

import os
import sys
import math
import numpy as np

criterion = nn.CrossEntropyLoss()
criterion_self = nn.MSELoss()

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def train(model, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, in_shape):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out0, out1, _, _ = model(inputs)               # Forward Propagation

        loss0 = criterion(out0, targets)  # Loss
        loss1 = criterion(out1, targets)
        self_loss = criterion_self(out0,out1)

        # print(self_loss*0.001)

        loss = loss0 + loss1 + self_loss*0.001

        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(out0.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.data[0], 100.*correct/total))
        sys.stdout.flush()


def test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct0 = 0
    correct1 = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out0, out1, _, _ = model(inputs)
        loss0 = criterion(out0, targets)  # Loss
        loss1 = criterion(out1, targets)

        # self_loss = criterion_self(out0,out1)
        # print("self_loss: ", self_loss)

        loss = loss0 + loss1

        out_mean = out0*0.5 + out1*0.5

        test_loss += loss.data[0]
        _, predicted = torch.max(out_mean.data, 1)
        _, predicted0 = torch.max(out0.data, 1)
        _, predicted1 = torch.max(out1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct0 += predicted0.eq(targets.data).cpu().sum()
        correct1 += predicted1.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*float(correct)/float(total)
    acc0 = 100.*float(correct0)/float(total)
    acc1 = 100.*float(correct1)/float(total)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%  Acc0@1: %.2f%%  Acc1@1: %.2f%%" %(epoch, loss.data[0], acc, acc0, acc1))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
                'model': model if use_cuda else model,
                'acc': acc,
                'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+fname+'.t7')
        best_acc = acc
    return best_acc

def extract_feat(model, trainloader, testloader, save_name):
    model.eval()
    number = 0
    for idx, (input, target) in enumerate(trainloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output
        _, _, output_32x32_0, output_32x32_1 = model(input_var)
        # print("output shape: ", output.shape)
        # print("output_32x32 shape: ", output_32x32.shape)
        # print("out_bij shape: ", out_bij.shape)

        output_32x32_0_cpu = output_32x32_0.cpu().detach().numpy()
        output_32x32_1_cpu = output_32x32_1.cpu().detach().numpy()

        target_cpu = target.cpu().detach().numpy()

        batch_size = output_32x32_0_cpu.shape[0]
        # print(batch_size)
        for i in range(batch_size):
            feat0 = output_32x32_0_cpu[i,:,:,:]
            feat1 = output_32x32_1_cpu[i,:,:,:]
            
            label = target_cpu[i]

            save_folder_0 = 'data/' + save_name + '_0'
            save_folder_1 = 'data/' + save_name + '_1'
            if(os.path.exists(save_folder_0)==False):
                os.mkdir(save_folder_0)
            if(os.path.exists(save_folder_1)==False):
                os.mkdir(save_folder_1)

            save_path_0 = save_folder_0 + '/' + str(label) + '/'
            save_path_1 = save_folder_1 + '/' + str(label) + '/'        
            if(os.path.exists(save_path_0)==False):
                os.mkdir(save_path_0)
            if(os.path.exists(save_path_1)==False):
                os.mkdir(save_path_1)

            np.save(save_path_0 + str(number) + '.npy',feat0)
            np.save(save_path_1 + str(number) + '.npy',feat1)

            number+=1
        
        sys.stdout.write('\r')
        sys.stdout.write('| Iter %3d' % (idx+1))
        sys.stdout.flush()


    number = 0
    for idx, (input, target) in enumerate(testloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output
        _, _, output_32x32_0, output_32x32_1 = model(input_var)
        # print("output shape: ", output.shape)
        # print("output_32x32 shape: ", output_32x32.shape)
        # print("out_bij shape: ", out_bij.shape)

        output_32x32_0_cpu = output_32x32_0.cpu().detach().numpy()
        output_32x32_1_cpu = output_32x32_1.cpu().detach().numpy()

        target_cpu = target.cpu().detach().numpy()

        batch_size = output_32x32_0_cpu.shape[0]
        # print(batch_size)
        for i in range(batch_size):
            feat0 = output_32x32_0_cpu[i,:,:,:]
            feat1 = output_32x32_1_cpu[i,:,:,:]
            
            label = target_cpu[i]

            save_folder_0 = 'data/' + save_name + '_0_val'
            save_folder_1 = 'data/' + save_name + '_1_val'
            if(os.path.exists(save_folder_0)==False):
                os.mkdir(save_folder_0)
            if(os.path.exists(save_folder_1)==False):
                os.mkdir(save_folder_1)

            save_path_0 = save_folder_0 + '/' + str(label) + '/'
            save_path_1 = save_folder_1 + '/' + str(label) + '/'        
            if(os.path.exists(save_path_0)==False):
                os.mkdir(save_path_0)
            if(os.path.exists(save_path_1)==False):
                os.mkdir(save_path_1)

            np.save(save_path_0 + str(number) + '.npy',feat0)
            np.save(save_path_1 + str(number) + '.npy',feat1)

            number+=1
        sys.stdout.write('\r')
        sys.stdout.write('| Iter %3d' % (idx+1))
        sys.stdout.flush()
