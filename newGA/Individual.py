import numpy as np
import sys
import os

dir=os.path.abspath("..")
sys.path.append(dir)

from newGA.ResNeXtUnit import ResNeXtUnit
from newGA.FirstConvUnit import FirstConvUnit
from newNetBlock.ResNeXtBlock import ResNeXtBlock
from Tool.utils import cal_indi_parameters_and_flops
class Individual():
    """个体代表网络结构"""
    def __init__(self,params,indi_id):
        """初始化个体

        Args:
            params: 配置参数字典
            indi_id: 个体编号（字符串或整数）
        """
        self.id=indi_id #个体的编号
        self.ntk=-1.0 # 个体的NTK
        self.acc=-1.0 # 个体的精度
        self.spantime=0 #个体在种群中的寿命
        self.units=[] #个体的具体单元组成
        self.unit_id=0 #个体中最后一个单元的编号
        self.params=params
        self.unit_length=np.random.randint(params["min_unit_length"],params["max_unit_length"]+1) #个体的长度
        # self.unit_length=params["max_unit_length"] #个体的长度
        self.image_channel=params["image_channel"] # 输入图片的通道数
        self.in_size=params["in_size"] # 输入图片的尺寸
        self.out_channel=params["out_channel"] # 经过第一个卷积层输出的通道数
        self.total_SENet_amount=np.random.randint(params["min_total_SENet_amount"],params["max_total_SENet_amount"]+1) # 个体中SENet的最大数量（SENet涉及参数量，参数量不宜过多）
        self.max_unit_SENet_amount=self.total_SENet_amount//(self.unit_length-1) # 一个unit中SENet的最大数量
        self.max_ResNeXt_amount=params["max_ResNeXt_amount"] # 一个unit中包含ResNeXt block的最大数量
        self.min_ResNeXt_amount=params["min_ResNeXt_amount"] # 一个unit中包含ResNeXt block的最小数量
        self.cardinality=params["cardinality"][:params["init_c_and_gw_index"]] # 一个包含所有cardinality的列表
        self.group_width=params["group_width"][:params["init_c_and_gw_index"]] # 一个包含所有group_width的列表
        # print(self.cardinality)
        # print(self.group_width)
        self.cardinality_with_group_width_up_limit=params["cardinality_with_group_width_up_limit"] #cardinality*group_width的上限 
        self.cardinality_with_group_width_down_limit=params["cardinality_with_group_width_down_limit"] #cardinality*group_width的下限
        self.exist_senet=0 #个体中的SENet数量
        self.conv_length=0 #个体中conv的数量
        self.parameters=0 #个体的参数量
        self.flops=0 #个体的计算量
    def reset_ntk(self):
        """重置个体的 NTK"""
        self.ntk=-1.0
    
    def reset_acc(self):
        """重置个体的精度"""
        self.acc=-1.0

    def reset(self):
        """重置个体的寿命与指标（spantime/ntk/acc）"""
        self.spantime=0
        self.ntk=-1.0
        self.acc=-1.0

    def init_first_conv_unit(self,unit_id,in_size=32,in_channel=3,out_channel=128):
        """初始化首个卷积单元"""
        self.unit_id+=1
        return FirstConvUnit(unit_id,in_size,in_channel,out_channel)

    def init_resnext_unit(self,unit_id,in_channel=128,stride=1):
        """初始化 ResNeXt 单元

        Args:
            unit_id: 单元编号
            in_channel: 输入通道
            stride: 首个 block 的 stride（后续 block 为 1）
        Returns:
            ResNeXtUnit
        """
        self.unit_id+=1
        block_amount=np.random.randint(self.min_ResNeXt_amount,self.max_ResNeXt_amount+1) #unit中block的数量
        cardinalitys,group_widths=self.get_cardinalitys_and_group_widths(self.cardinality,self.group_width,block_amount) #获取随机生成的cardinalitys和group_widths
        # 计算一个unit中的SENet数量，如果unit的block数量小于等于unit最大的SENet数量，则senet_amount=block_amount,否则senet_amount=self.max_unit_SENet_amount
        if block_amount>self.max_unit_SENet_amount:
            senet_amount=np.random.randint(self.max_unit_SENet_amount+1) 
        else:
            senet_amount=np.random.randint(block_amount+1)
        hasSENets=[False for _ in range(block_amount)] #随机初始化每个ResNeXt Block是否含有SENet
        # 选择senet_amount个block初始化带有SENet
        indexs=np.random.choice(list(range(block_amount)),senet_amount,replace=False)
        self.exist_senet+=senet_amount
        for index in indexs:
            hasSENets[index]=True
        out_channel=ResNeXtBlock.expansion*cardinalitys[-1]*group_widths[-1]
        return ResNeXtUnit(unit_id,block_amount,in_channel,out_channel,cardinalitys,group_widths,hasSENets,stride)
            
    def initialize(self):
        """生成完整个体：首卷积 + 多个 ResNeXt 单元"""
        first_conv_unit=self.init_first_conv_unit(self.unit_id,self.in_size,self.image_channel,self.out_channel)
        self.units.append(first_conv_unit)
        in_channel=first_conv_unit.out_channel
        
        # 设置unit中的stride,总共只有2个unit的stride为2
        strides=[1]*(self.unit_length-1)
        indexs=np.random.choice(list(range(self.unit_length-1)),2,replace=False)
        for index in indexs:
            strides[index]=2
        # 生成ResNeXTUnit列表
        for i in range(self.unit_length-1):
            resnext_unit=self.init_resnext_unit(self.unit_id,in_channel,strides[i])
            self.units.append(resnext_unit)
            in_channel=resnext_unit.out_channel
        self.cal_conv_length_and_senet()
    

    def get_cardinalitys_and_group_widths(self,cardinalitys,group_widths,block_amount):
        """生成一个 ResNeXtUnit 中每个 Block 的 `(cardinality, group_width)` 随机组合（受上下限约束）"""
        
        block_cardinalitys=[]
        block_group_widths=[]
        cnum=len(self.cardinality)
        gwnum=len(self.group_width)
        for i in range(block_amount):
            cardinality_index=np.random.choice(cnum)
            group_width_index=np.random.choice(gwnum)
            while(cardinalitys[cardinality_index]*group_widths[group_width_index]>self.cardinality_with_group_width_up_limit or cardinalitys[cardinality_index]*group_widths[group_width_index]<self.cardinality_with_group_width_down_limit):
                #TODO 这种选择会使得64和32出现的概率变低
                cardinality_index=np.random.choice(cnum)
                group_width_index=np.random.choice(gwnum)
            block_cardinalitys.append(cardinalitys[cardinality_index])
            block_group_widths.append(group_widths[group_width_index])
        return (block_cardinalitys,block_group_widths)
    
    def print_individual(self):
        print("id:{},ResNeXt_unit_amount:{},ntk:{},acc:{}".format(self.id,self.unit_length,self.ntk,self.acc))

    def print_ntk(self):
        print("ntk:{}".format(self.ntk))

    def print_acc(self):
        print("acc:{}".format(self.acc))

    def cal_conv_length_and_senet(self):
        """统计个体的卷积层数量与 SENet 数量"""
        length=1
        senet=0
        for unit in self.units[1:]:
            length+=unit.block_amount*3
            senet+=sum(unit.hasSENets)
        self.conv_length=length
        self.exist_senet=senet


        
    def __str__(self):
        return "[id:{},unit_amount:{},conv_length:{},senet_amount:{},spantime:{},ntk:{},acc:{},params:{},FLOPs:{}]".format(self.id,self.unit_length,self.conv_length,self.exist_senet,self.spantime,self.ntk,self.acc,self.parameters,self.flops)
        # return "[id:{},unit_amount:{},conv_length:{},senet_amount:{},ntk:{},spantime:{},acc:{}]".format(self.id,self.unit_length,0,self.exist_senet,self.ntk,self.spantime,self.acc)




if __name__=="__main__":
    from newSetting.config import Config as config
    params=config.dev_params
    for _ in range(40):
        indi=Individual(params,0)
        indi.initialize()
        cal_indi_parameters_and_flops(indi)
        print(indi)
        # for unit in indi.units:
            # print(unit)
"""个体定义：一条可搜索的网络结构

说明
- 个体由一个首卷积单元 + 若干 ResNeXt 单元构成
- ResNeXt 单元内部包含多个 block，每个 block 具有 `(cardinality, group_width)` 配置，可能包含 SENet
- 个体同时维护统计指标（NTK/ACC/寿命/参数量/FLOPs）与结构统计（conv 数量、SENet 数量）
"""
