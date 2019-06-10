import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from trainer_encryptor import Trainer
import tensorboardX
import logging
import argparse
from utils import get_config, write_loss, prepare_sub_folder
import os
import shutil

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/example.yaml', help='Path to the config file.')
    opts = parser.parse_args()

    config = get_config(opts.config)
    max_epoch = config['max_epoch']

    m_trainer = Trainer(logger,config)
    m_trainer.to(device)

    train_writer = tensorboardX.SummaryWriter(os.path.join("logs/", config['exp_name']))
    output_directory = os.path.join("Expriments/", config['exp_name'])
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

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
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    for epoch in range(1,max_epoch+1):
        m_trainer.train_epoch(epoch,trainloader,train_writer,image_directory)
        m_trainer.test_epoch(epoch,testloader,train_writer,image_directory,checkpoint_directory)