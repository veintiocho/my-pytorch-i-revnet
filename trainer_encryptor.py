import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import weights_init, get_config, write_loss, write_ori_images, write_BNrecon_images, write_recon_images, prepare_sub_folder
import os
import sys
import argparse
from PIL import Image
from base_models import *
import shutil

from models import iRevEncryptor

class Trainer(nn.Module):
    def __init__(self,serverLogger,hyperparameters):
        super(Trainer, self).__init__()
        self.logger = serverLogger
        self.lr = hyperparameters['lr']
        self.device = torch.device('cuda')

        self.encryptor = iRevEncryptor.iRevEncryptor(nClasses=hyperparameters['classCount'])
        guide_net_name = hyperparameters['guide_net']

        print('==> Building guide model:',hyperparameters['guide_net'])
        if guide_net_name == 'vgg':
            self.guide_net = VGG('VGG19',32)
        elif guide_net_name == 'ResNet50':
            self.guide_net = ResNet50(32)
        elif guide_net_name == 'PreActResNet18':
            self.guide_net = PreActResNet18()
        elif guide_net_name == 'GoogLeNet':
            self.guide_net = GoogLeNet()
        elif guide_net_name == 'DenseNet121':
            self.guide_net = DenseNet121()
        elif guide_net_name == 'ResNeXt29_2x64d':
            self.guide_net = ResNeXt29_2x64d()
        elif guide_net_name == 'MobileNet':
            self.guide_net = MobileNet()
        elif guide_net_name == 'DPN92':
            self.guide_net = DPN92()
        elif guide_net_name == 'ShuffleNetG2':
            self.guide_net = ShuffleNetG2()
        elif guide_net_name == 'SENet18':
            self.guide_net = SENet18(3)
        else:
            self.logger.error("Unsupported base model!!!")

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.iterCount = 0
        self.best_acc = 0

    def adjust_learning_rate(self, epoch, max_epoch):
        optim_factor = 0
        p_ratio = float(epoch)/float(max_epoch)
        if(p_ratio > 0.8):
            optim_factor = 3
        elif(p_ratio > 0.6):
            optim_factor = 2
        elif(p_ratio > 0.3):
            optim_factor = 1
        lr = init*math.pow(0.2, optim_factor)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def write_loss(self,train_writer):
        members = [attr for attr in dir(self) \
               if not callable(getattr(self, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
        for m in members:
            train_writer.add_scalar(m, getattr(self, m), self.iterCount + 1)

    def train_epoch(self,epoch,trainloader,train_writer,image_directory):
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            self.iterCount += 1
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            feat_ori, feat_in = self.encryptor(inputs)

            recon_images = self.encryptor.inverse(feat_ori)
            in_recon_images = self.encryptor.inverse(feat_in)
            # in_recon_images = self.encryptor.inverse(feat_in)

            outputs = self.guide_net(feat_ori)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.train_loss_record = train_loss/(batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if self.iterCount%20 == 0:
                self.write_loss(train_writer)

            if self.iterCount%200 == 0:
                write_recon_images(recon_images, 16, image_directory, 'train_%08d' % (self.iterCount + 1))
                write_ori_images(inputs, 16, image_directory, 'train_%08d' % (self.iterCount + 1))
                write_BNrecon_images(in_recon_images,16,image_directory, 'train_%08d' % (self.iterCount + 1))
            
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d]\t\tLoss: %.4f Acc@1: %.3f%% (%d/%d)' % (epoch, (train_loss/(batch_idx+1)), 100.*correct/total, correct, total))
            sys.stdout.flush()
        print('\n')

    def test_epoch(self,epoch,testloader,test_writer,image_directory,checkpoint_directory):
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            feat_ori, feat_in = self.encryptor(inputs)
            recon_images = self.encryptor.inverse(feat_ori)
            in_recon_images = self.encryptor.inverse(feat_in)

            outputs = self.guide_net(feat_ori)

            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            self.test_loss_record = test_loss/(batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sys.stdout.write('\r')
            sys.stdout.write('| Iter [%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%% (%d/%d)' % (batch_idx, len(testloader), (test_loss/(batch_idx+1)), 100.*correct/total, correct, total))
            sys.stdout.flush()

            self.write_loss(test_writer)
            write_recon_images(recon_images, 16, image_directory, 'test_%08d' % (self.iterCount + 1))
            write_ori_images(inputs, 16, image_directory, 'test_%08d' % (self.iterCount + 1))
            write_BNrecon_images(in_recon_images,16,image_directory, 'test_%08d' % (self.iterCount + 1))
            acc = 100.*correct/total

        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(checkpoint_directory):
                os.mkdir(checkpoint_directory)
            torch.save(state, checkpoint_directory + '/ckpt-' + str(int(acc)) + '.t7')
            self.best_acc = acc

        print('\n')
