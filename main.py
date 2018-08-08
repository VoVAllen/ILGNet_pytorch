# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

# import lrs
import tensorboardX

from ILGNet import ILGNet
from data_loader import AVADataset

from tensorboardX import SummaryWriter


def getName(prefix):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(prefix, current_time + '_' + socket.gethostname())
    return log_dir


writer = SummaryWriter(getName("/data/output/"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_acc(label: torch.Tensor, pred: torch.Tensor):
    dist = torch.arange(10).float().to(device)
    l_mean = (label.view(-1, 10) * dist).sum(dim=1)
    l_good = l_mean > 5
    acc = (pred.argmax(dim=1).byte() == l_good).float().mean()
    return acc


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Scale(260),
        transforms.RandomCrop(227),
        transforms.ToTensor()])

    valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
                                             shuffle=False, num_workers=config.num_workers)

    model = ILGNet()
    model.load_state_dict(torch.load('/data/jinjing/ILGNet_pytorch.pth'))
    model = model.to(device)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    step = 0
    # do validation after each epoch
    batch_val_losses = []
    val_acc = []
    for i, data in enumerate(val_loader):
        print(i)
        model.eval()
        images = data['image'].to(device)
        labels = data['annotations'].to(device).float()
        with torch.no_grad():
            outputs = model(images)
        step += 1
        val_acc.append(compute_acc(labels, outputs))

    writer.add_scalar('val/accuracy', np.mean(val_acc), step)
    print(np.mean(val_acc))
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/data/full_ava/train/images/')
    parser.add_argument('--val_img_path', type=str, default='/data/full_ava/train/images/')
    parser.add_argument('--test_img_path', type=str, default='/data/full_ava/train/images/')
    parser.add_argument('--train_csv_file', type=str, default='/data/full_ava/train/train.csv')
    parser.add_argument('--val_csv_file', type=str, default='/data/full_ava/train/val.csv')
    parser.add_argument('--test_csv_file', type=str, default='/data/full_ava/train/test.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--conv_base_lr', type=float, default=1e-3)
    parser.add_argument('--dense_lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='/data/output/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    # config = parser.parse_args()
    config, unknown = parser.parse_known_args()
    writer.add_text("Config", str(config))

    main(config)
