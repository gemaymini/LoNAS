import torch.nn as nn
import torch
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    expansion=2
    def __init__(self,in_channel,cardinality,bottleneck_width,stride,hasSENet):
        super(ResNeXtBlock,self).__init__()
        group_width=cardinality*bottleneck_width
        self.conv1=nn.Conv2d(in_channel,group_width,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(group_width)
        self.conv2=nn.Conv2d(group_width,group_width,kernel_size=3,stride=stride,padding=1,groups=cardinality,bias=False)
        self.bn2=nn.BatchNorm2d(group_width)
        self.conv3=nn.Conv2d(group_width,self.expansion*group_width,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(self.expansion*group_width)
        self.hasSENet=hasSENet

        self.shortcut=nn.Sequential()
        if stride!=1 or in_channel!=self.expansion*group_width:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channel,self.expansion*group_width,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
        if hasSENet:
            self.fc1=nn.Conv2d(self.expansion*group_width,self.expansion*group_width//16,kernel_size=1)
            self.fc2=nn.Conv2d(self.expansion*group_width//16,self.expansion*group_width,kernel_size=1)

        
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        if self.hasSENet:
            # Squeeze
            w=F.avg_pool2d(out,out.size(2))
            w=F.relu(self.fc1(w))
            w=F.sigmoid(self.fc2(w))
            # Excitation
            out=out*w
        out+=self.shortcut(x)
        out=F.relu(out)
        return out


