import itertools
import os
import argparse
import random
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from data_feeder import ASVDataSet, load_data, collate_fn_pad
from torch.utils.data import DataLoader
import numpy as np
import torchaudio
import math
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn import Parameter
from tqdm import tqdm
import soundfile as sf
from feature_extract import extract_after_enhance


def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,        # torch.Size([70, 1, 129])
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Interaction(nn.Module):
    def __init__(self, in_planes, out):
        super(Interaction, self).__init__()
        
        self.conv_fusion = nn.Conv2d(in_planes, out, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_sigmoid = nn.Conv2d(in_planes, out, kernel_size=3, stride=1, padding=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, enh, ori):
        cat = torch.cat((enh, ori), 1)
        fusion = self.conv_fusion(cat)
        w = self.sig(self.conv_sigmoid(cat))        # torch.Size([2, 16, 108, 150])
        ori = fusion + w * ori
        enh = fusion + w * enh
        return enh, ori

class ResNet(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
    def __init__(self, block, layers, num_classes):

        self.inplanes = 16 

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        # self.classifier = nn.Linear(128 * block.expansion, num_classes)
        self.classifier = AngleLinear(128*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ## print(x.size())

        x = self.layer1(x)
        ## print(x.size())
        x = self.layer2(x)
        ## print(x.size())
        x = self.layer3(x)
        ## print(x.size())
        x = self.layer4(x)
        ## print(x.size())

        x = self.avgpool(x).view(x.size()[0], -1)
        ## print(x.shape)
        out = self.classifier(x)
        ## print(out.shape)

        # return F.log_softmax(out, dim=-1)
        return out

class ResNet_fusion(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
    def __init__(self, block, layers, num_classes):
        
        self.inplanes = 16 
        self.inplanes1 = 16
        super(ResNet_fusion, self).__init__()


        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv11 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn11 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.interaction1 = Interaction(32, 16)
        self.interaction2 = Interaction(32, 16)
        self.interaction3 = Interaction(64, 32)
        self.interaction4 = Interaction(128, 64)
        self.interaction5 = Interaction(256, 128)

        self.SpatialAttention = SpatialAttention()


        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

      

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.classifier = nn.Linear(128 * block.expansion, num_classes)
        self.classifier = AngleLinear(128*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.sigmoid = nn.Sigmoid()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        enh = x[:, :1, :,:]        # torch.Size([2, 1, 432, 600])
        
        ori = x[:, 1:2, :,:]


        enh = self.conv1(enh)       # torch.Size([2, 16, 216, 300])
        enh = self.bn1(enh)
        enh = self.relu(enh)
        enh = self.maxpool(enh)     # torch.Size([2, 16, 108, 150])


        ori = self.conv11(ori)
        ori = self.bn11(ori)
        ori = self.relu1(ori)
        ori = self.maxpool1(ori)

        enh, ori = self.interaction1(enh, ori)
        enh = (1 - self.SpatialAttention(ori)) * enh + self.SpatialAttention(ori) * ori
    
        enh = self.layer1(enh)      # torch.Size([2, 16, 108, 150])
        enh = self.layer2(enh)      # torch.Size([2, 32, 54, 75])
        enh = self.layer3(enh)      # torch.Size([2, 64, 54, 75])
        enh = self.layer4(enh)      # torch.Size([2, 128, 27, 38])
        x = self.avgpool(enh).view(enh.size()[0], -1)
        #print(x.shape)
        out = self.classifier(x)
        #print(out.shape)

        # return F.log_softmax(out, dim=-1)
        return out

def se_resnet34_fusion(**kwargs):
    model = ResNet_fusion(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        
        self.lstm_layer = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def forward(self, x):
        self.lstm_layer.flatten_parameters()

        e_1 = self.conv_block_1(x)  
        e_2 = self.conv_block_2(e_1)    
        e_3 = self.conv_block_3(e_2)    
        e_4 = self.conv_block_4(e_3)   
        e_5 = self.conv_block_5(e_4)  

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]     ----------[2, 800, 3072]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]     

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))  # [2, 128, 9, 200]
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))      # [2, 64, 19, 200]
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))      # [2, 32, 39, 200]
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))      # [2, 16, 80, 200]
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))      # [2, 1, 161, 200]

        return d_5
    

class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])



if __name__ == '__main__':
    se_resnet34 = se_resnet34(num_classes=2)
    se_resnet34_fusion = se_resnet34_fusion(num_classes=2)
    CRN = CRN()
    interaction1 = Interaction(32, 16)
    num_params_resnet34 = sum(i.numel() for i in se_resnet34.parameters() if i.requires_grad)  # 1343760  1.34M
    num_params_resnet34_fusion = sum(i.numel() for i in se_resnet34_fusion.parameters() if i.requires_grad)  # 2137250 2.14M
    num_params_CRN = sum(i.numel() for i in CRN.parameters() if i.requires_grad)  # 17579459 17.58M
    num_params_interaction1 = sum(i.numel() for i in interaction1.parameters() if i.requires_grad)

    x1 = torch.randn(2, 1, 432, 600)
    x11 = torch.randn(2, 2, 432, 600)
    x2 = torch.randn(2, 1, 161, 200)
    y1 = se_resnet34(x1)
    y2 = se_resnet34_fusion(x11)
    y3 = CRN(x2)

    # Number of learnable params: se_resnet34 1343760, se_resnet34_fusion 2137250, CRN: 17579459, num_params_interaction1: 9216.
    print('Number of learnable params: se_resnet34 {}, se_resnet34_fusion {}, CRN: {}, num_params_interaction1: {}.'.format(num_params_resnet34, num_params_resnet34_fusion, num_params_CRN, num_params_interaction1))
