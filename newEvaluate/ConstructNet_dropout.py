import sys
import os
dir=os.path.abspath("..")
sys.path.append(dir)

from newNetBlock.FirstConvBlock import FirstConvBlock
from newNetBlock.ResNeXtBlock import ResNeXtBlock
import torch as t
import torch.nn as nn
import torch.nn.functional as F



class ConstructNet(nn.Module):
    def __init__(self,indi,num_classes=10):
        super(ConstructNet,self).__init__()
        # self.names = self.__dict__
        self.units=indi.units
        self.num_classes=num_classes
        self.first_conv=FirstConvBlock(self.units[0].in_channel,self.units[0].out_channel)
        self.in_channel=self.units[0].out_channel
        self.resnet_layers=self._make_conv_sequential()
        self.fc=nn.Linear(self.units[-1].out_channel,self.num_classes)

    def _make_conv_sequential(self):
        """将所有 ResNeXtUnit 转成顺序层组"""
        layer=[]
        firstConvUnit=self.units[0]
        for unit in self.units[1:]:
            layer.append(self._make_resnext_layer(unit))
        return nn.Sequential(*layer)

    def _make_resnext_layer(self,unit):
        """构建单个 ResNeXtUnit 的层序列，首个 block 采用给定 stride"""
        strides=[unit.stride]+[1]*(unit.block_amount-1)
        layers=[]
        for index,stride in enumerate(strides):
            layers.append(ResNeXtBlock(self.in_channel,unit.cardinalitys[index],unit.group_widths[index],stride,unit.hasSENets[index]))
            self.in_channel=unit.cardinalitys[index]*unit.group_widths[index]*ResNeXtBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out=self.first_conv(x)
        out=self.resnet_layers(out)
        out=F.avg_pool2d(out,8)
        out=out.view(out.size(0),-1)
        return self.fc(F.dropout(out,p=0.5))


if __name__=="__main__":
    from newGA.Individual import Individual
    from newSetting.config import Config
    import numpy as np
    from newEvaluate.EvolveDataLoader import Evolve_DataLoader
    trainloader,validloader=Evolve_DataLoader.data_loader()
    
    input=t.randn(1,3,32,32)
    indi=Individual(Config.dev_params,0)
    indi.initialize()
    for unit in indi.units:
        print(unit)
    net=ConstructNet(indi,10)
    print(net)
    output=net(input)
"""构建含 Dropout 的网络（CIFAR 版）"""
