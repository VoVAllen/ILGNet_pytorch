import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from collections import OrderedDict


class Inception3a(nn.Module):
    def __init__(self, prefix="inception_3a"):
        super(Inception3a, self).__init__()
        self.conv1x1 = nn.Sequential(OrderedDict([
            (prefix + '/1x1', nn.Conv2d(192, 64, kernel_size=1)),
            (prefix + '/relu_1x1', nn.ReLU())
        ]))
        self.conv3x3_reduce = nn.Sequential(OrderedDict([
            (prefix + '/3x3_reduce', nn.Conv2d(192, 96, kernel_size=1)),
            (prefix + '/relu_3x3_reduce', nn.ReLU())
        ]))
        self.conv3x3 = nn.Sequential(OrderedDict([
            (prefix + '/3x3', nn.Conv2d(96, 128, padding=1, kernel_size=3)),
            (prefix + '/relu_3x3', nn.ReLU())
        ]))
        self.conv5x5_reduce = nn.Sequential(OrderedDict([
            (prefix + '/5x5_reduce', nn.Conv2d(192, 16, kernel_size=1)),
            (prefix + '/relu_5x5_reduce', nn.ReLU())
        ]))
        self.conv5x5 = nn.Sequential(OrderedDict([
            (prefix + '/5x5', nn.Conv2d(16, 32, padding=2, kernel_size=5)),
            (prefix + '/relu_5x5', nn.ReLU())
        ]))
        self.pool = nn.Sequential(OrderedDict([
            (prefix + '/pool', nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True))
        ]))
        self.pool_proj = nn.Sequential(OrderedDict([
            (prefix + '/pool_proj', nn.Conv2d(192, 32, kernel_size=1)),
            (prefix + '/relu_pool_proj', nn.ReLU())
        ]))

    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(self.conv3x3_reduce(x))
        branch3 = self.conv5x5(self.conv5x5_reduce(x))
        branch4 = self.pool_proj(self.pool(x))
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class Inception3b(nn.Module):
    def __init__(self, prefix="inception_3b"):
        super(Inception3b, self).__init__()
        self.conv1x1 = nn.Sequential(OrderedDict([
            (prefix + '/1x1', nn.Conv2d(256, 128, kernel_size=1)),
            (prefix + '/relu_1x1', nn.ReLU())
        ]))
        self.conv3x3_reduce = nn.Sequential(OrderedDict([
            (prefix + '/3x3_reduce', nn.Conv2d(256, 128, kernel_size=1)),
            (prefix + '/relu_3x3_reduce', nn.ReLU())
        ]))
        self.conv3x3 = nn.Sequential(OrderedDict([
            (prefix + '/3x3', nn.Conv2d(128, 192, padding=1, kernel_size=3)),
            (prefix + '/relu_3x3', nn.ReLU())
        ]))
        self.conv5x5_reduce = nn.Sequential(OrderedDict([
            (prefix + '/5x5_reduce', nn.Conv2d(256, 32, kernel_size=1)),
            (prefix + '/relu_5x5_reduce', nn.ReLU())
        ]))
        self.conv5x5 = nn.Sequential(OrderedDict([
            (prefix + '/5x5', nn.Conv2d(32, 96, padding=2, kernel_size=5)),
            (prefix + '/relu_5x5', nn.ReLU())
        ]))
        self.pool = nn.Sequential(OrderedDict([
            (prefix + '/pool', nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True))
        ]))
        self.pool_proj = nn.Sequential(OrderedDict([
            (prefix + '/pool_proj', nn.Conv2d(256, 64, kernel_size=1)),
            (prefix + '/relu_pool_proj', nn.ReLU())
        ]))


    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(self.conv3x3_reduce(x))
        branch3 = self.conv5x5(self.conv5x5_reduce(x))
        branch4 = self.pool_proj(self.pool(x))
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class Inception4a(nn.Module):
    def __init__(self, prefix="inception_4a"):
        super(Inception4a, self).__init__()
        self.conv1x1 = nn.Sequential(OrderedDict([
            (prefix + '/1x1', nn.Conv2d(480, 192, kernel_size=1)),
            (prefix + '/relu_1x1', nn.ReLU())
        ]))
        self.conv3x3_reduce = nn.Sequential(OrderedDict([
            (prefix + '/3x3_reduce', nn.Conv2d(480, 96, kernel_size=1)),
            (prefix + '/relu_3x3_reduce', nn.ReLU())
        ]))
        self.conv3x3 = nn.Sequential(OrderedDict([
            (prefix + '/3x3', nn.Conv2d(96, 208, padding=1, kernel_size=3)),
            (prefix + '/relu_3x3', nn.ReLU())
        ]))
        self.conv5x5_reduce = nn.Sequential(OrderedDict([
            (prefix + '/5x5_reduce', nn.Conv2d(480, 16, kernel_size=1)),
            (prefix + '/relu_5x5_reduce', nn.ReLU())
        ]))
        self.conv5x5 = nn.Sequential(OrderedDict([
            (prefix + '/5x5', nn.Conv2d(16, 48, padding=2, kernel_size=5)),
            (prefix + '/relu_5x5', nn.ReLU())
        ]))
        self.pool = nn.Sequential(OrderedDict([
            (prefix + '/pool', nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True))
        ]))
        self.pool_proj = nn.Sequential(OrderedDict([
            (prefix + '/pool_proj', nn.Conv2d(480, 64, kernel_size=1)),
            (prefix + '/relu_pool_proj', nn.ReLU())
        ]))


    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(self.conv3x3_reduce(x))
        branch3 = self.conv5x5(self.conv5x5_reduce(x))
        branch4 = self.pool_proj(self.pool(x))
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ILGNet(nn.Module):
    def __init__(self):
        super(ILGNet, self).__init__()
        self.feature = nn.Sequential(
            OrderedDict([
                ('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)),
                ('conv1/relu_7x7', nn.ReLU()),
                ('pool1/3x3_s2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                ('pool1/norm1', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)),
                ('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1)),
                ('conv2/relu_3x3_reduce', nn.ReLU()),
                ('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, padding=1)),
                ('conv2/relu_3x3', nn.ReLU()),
                ('conv2/norm2', nn.LocalResponseNorm(size=5, alpha=0.00001, beta=0.75)),
                ('pool2/3x3_s2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            ])
        )

        self.pool3 = nn.Sequential(OrderedDict([
            ('pool3/3x3_s2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        ]))
        self.inception_3a = Inception3a()
        self.inception_3b = Inception3b()
        self.inception_4a = Inception4a()

        self.loss1_ave_pool = nn.Sequential(OrderedDict([
            ('loss1/ave_pool', nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True))
        ]))

        self.loss1_conv = nn.Sequential(OrderedDict([
            ('loss1/conv', nn.Conv2d(512, 128, kernel_size=1)),
            ('loss1/relu_conv', nn.ReLU())
        ]))

        self.temp1 = nn.Sequential(OrderedDict([
            ('temp1', nn.Linear(256 * 28 * 28, 256)),
            ('temp1_relu', nn.ReLU()),
            ('temp1_drop', nn.Dropout(0.7))
        ]))

        self.temp1 = nn.Sequential(OrderedDict([
            ('temp1', nn.Linear(256 * 28 * 28, 256)),
            ('temp1_relu', nn.ReLU()),
            ('temp1_drop', nn.Dropout(0.7))
        ]))

        self.temp2 = nn.Sequential(OrderedDict([
            ('temp2', nn.Linear(480 * 28 * 28, 256)),
            ('temp2_relu', nn.ReLU()),
            ('temp2_drop', nn.Dropout(0.7))
        ]))

        self.temp3 = nn.Sequential(OrderedDict([
            ('temp3', nn.Linear(128 * 4 * 4, 512)),
            ('temp3_relu', nn.ReLU()),
            ('temp3_drop', nn.Dropout(0.7))
        ]))

        self.temp_cjy = nn.Sequential(OrderedDict([
            ('temp_cjy', nn.Linear(1024, 1024)),
            ('temp_cjy_relu', nn.ReLU()),
            ('temp_cjy_drop', nn.Dropout(0.7))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('loss1/classifier_cjy', nn.Linear(1024, 2)),
        ]))

    def forward(self, x):
        pool2_3x3_s2_output = self.feature(x)
        inception_3a_output = self.inception_3a(pool2_3x3_s2_output)
        inception_3b_output = self.inception_3b(inception_3a_output)
        inception_4a_output = self.inception_4a(self.pool3(inception_3b_output))
        loss_ave_pool_output = self.loss1_ave_pool(inception_4a_output)
        loss1_relu_conv_output = self.loss1_conv(loss_ave_pool_output)
        flat_loss1_relu_conv_output = loss1_relu_conv_output.view(loss_ave_pool_output.size(0), -1)
        flat_inception_3a_output = inception_3a_output.view(inception_3a_output.size(0), -1)
        flat_inception_3b_output = inception_3b_output.view(inception_3b_output.size(0), -1)

        temp1_output = self.temp1(flat_inception_3a_output)
        temp2_output = self.temp2(flat_inception_3b_output)
        temp3_output = self.temp3(flat_loss1_relu_conv_output)

        temp_output = torch.cat([temp1_output, temp2_output, temp3_output], dim=1)

        temp_cjy_output = self.temp_cjy(temp_output)

        classfier_cjy = self.classifier(temp_cjy_output)

        return F.softmax(classfier_cjy, dim=1)
