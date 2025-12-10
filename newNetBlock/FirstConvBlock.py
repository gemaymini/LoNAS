import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FirstConvBlock,self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.bn=nn.BatchNorm2d(out_channel)
    
    def forward(self,x):
        return F.relu(self.bn(self.conv(x)))
"""首卷积块：将输入通道映射到指定输出通道

结构
- 1×1 卷积 + BN + ReLU
"""
