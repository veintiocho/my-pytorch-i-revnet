import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse

from models.utils_cifar import train, test, std, mean, get_hms, extract_feat
from models.iRevEncryptor import iRevEncryptor

import numpy as np

def extract_feat(model, trainloader, testloader, save_name):
    model.eval()
    number = 0
    for idx, (input, target) in enumerate(trainloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output
        out_origin, out_se = model(input_var)

        out_origin_cpu = out_origin.cpu().detach().numpy()
        out_se_cpu = out_se.cpu().detach().numpy()

        target_cpu = target.cpu().detach().numpy()

        batch_size = out_se.shape[0]
        # print(batch_size)
        for i in range(batch_size):
            feat_origin = out_origin_cpu[i,:,:,:]
            feat_se = out_se_cpu[i,:,:,:]
            
            label = target_cpu[i]

            save_folder_origin = 'data/' + save_name + '_origin'
            save_folder_se = 'data/' + save_name + '_se'
            if(os.path.exists(save_folder_origin)==False):
                os.mkdir(save_folder_origin)
            if(os.path.exists(save_folder_se)==False):
                os.mkdir(save_folder_se)

            save_path_origin = save_folder_origin + '/' + str(label) + '/'
            save_path_se = save_folder_se + '/' + str(label) + '/'        
            if(os.path.exists(save_path_origin)==False):
                os.mkdir(save_path_origin)
            if(os.path.exists(save_path_se)==False):
                os.mkdir(save_path_se)

            np.save(save_path_origin + str(number) + '.npy',feat_origin)
            np.save(save_path_se + str(number) + '.npy',feat_se)

            number+=1
        
        sys.stdout.write('\r')
        sys.stdout.write('| Iter %3d' % (idx+1))
        sys.stdout.flush()


    number = 0
    for idx, (input, target) in enumerate(testloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output
        out_origin, out_se = model(input_var)

        out_origin_cpu = out_origin.cpu().detach().numpy()
        out_se_cpu = out_se.cpu().detach().numpy()

        target_cpu = target.cpu().detach().numpy()

        batch_size = out_se.shape[0]
        # print(batch_size)
        for i in range(batch_size):
            feat_origin = out_origin_cpu[i,:,:,:]
            feat_se = out_se_cpu[i,:,:,:]
            
            label = target_cpu[i]

            save_folder_origin = 'data/' + save_name + '_origin_val'
            save_folder_se = 'data/' + save_name + '_se_val'
            if(os.path.exists(save_folder_origin)==False):
                os.mkdir(save_folder_origin)
            if(os.path.exists(save_folder_se)==False):
                os.mkdir(save_folder_se)

            save_path_origin = save_folder_origin + '/' + str(label) + '/'
            save_path_se = save_folder_se + '/' + str(label) + '/'        
            if(os.path.exists(save_path_origin)==False):
                os.mkdir(save_path_origin)
            if(os.path.exists(save_path_se)==False):
                os.mkdir(save_path_se)

            np.save(save_path_origin + str(number) + '.npy',feat_origin)
            np.save(save_path_se + str(number) + '.npy',feat_se)

            number+=1
        
        sys.stdout.write('\r')
        sys.stdout.write('| Iter %3d' % (idx+1))
        sys.stdout.flush()

def test_feat(model, dataloader):
    model.eval()
    number = 0
    for idx, (input, target) in enumerate(trainloader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output
        out_origin, out_se = model(input_var)
        print("out_origin shape: ", out_origin.shape)
        print("out_se shape: ", out_se.shape)


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/cyg/Documents/PaperProject/pytorch-i-revnet/checkpoint/enc-resnext-0_75-100/ckpt-71.t7',
                    type=str, help='Trained state_dict file path to open')
args = parser.parse_args()


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean['cifar100'], std['cifar100']),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean['cifar100'], std['cifar100']),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
nClasses = 100
in_shape = [3, 32, 32]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

model = iRevEncryptor(nClasses,nBlocks=[12], nStrides=[1], nChannels=[16])
model.load_weights(args.trained_model)
model.cuda()
model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
cudnn.benchmark = True

extract_feat(model,trainloader,testloader,'iRev_SE_feat')

