# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera

from builtins import super
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], 1)
        x = self.bn(x)
        return F.relu(x)
    

class NonBottleneck1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-3)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        output = self.conv3x1_1(x)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output + x)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))

        for _ in range(5):
           self.layers.append(NonBottleneck1d(64, 0.1, 1))  
        self.layers.append(DownsamplerBlock(64, 128))

        for _ in range(2):
            self.layers.append(NonBottleneck1d(128, 0.1, 2))
            self.layers.append(NonBottleneck1d(128, 0.1, 4))
            self.layers.append(NonBottleneck1d(128, 0.1, 8))
            self.layers.append(NonBottleneck1d(128, 0.1, 16))

        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, x, predict=False):
        x = self.initial_block(x)
        for layer in self.layers: x = layer(x)
        if predict: x = self.output_conv(x)
        return x


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(NonBottleneck1d(64, 0, 1))
        self.layers.append(NonBottleneck1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(NonBottleneck1d(16, 0, 1))
        self.layers.append(NonBottleneck1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        x = self.output_conv(x)
        return x


class LaneExist(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-3))

        self.layers_final = nn.ModuleList()
        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(3965, 128)
        self.linear2 = nn.Linear(128, 4)

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        x = F.relu(x)
        for layer in self.layers_final: x = layer(x)

        x = F.softmax(x, dim=1)
        x = self.maxpool(x)
        x = x.view(-1, 3965)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class ERFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)
        self.lane_exist = LaneExist(4)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder.forward(x), self.lane_exist(x)
