import sys
import os
import numpy as np
import random


dir=os.path.abspath("..")
sys.path.append(dir)

from newSetting.config import Config
from newGA.Individual import Individual
from newNetBlock.ResNeXtBlock import ResNeXtBlock
import numpy as np

if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

class doMutation():
    def __init__(self,indis,log):
        self.offsprings=indis
        self.log=log
        self.cardinalitys=params["cardinality"]
        self.group_widths=params["group_width"]
        # print(self.cardinalitys)
        self.cpos=list(range(len(self.cardinalitys)))
        self.gwpos=list(range(len(self.group_widths)))
        self.cardinality_with_group_width_up_limit=params["cardinality_with_group_width_up_limit"] #cardinality*group_width的上限 
        self.cardinality_with_group_width_down_limit=params["cardinality_with_group_width_down_limit"] #cardinality*group_width的下限

    def add_block(self,indi,unit_pos,block_pos):
        """在指定位置插入一个新的 block（随机合法的 `(cardinality, group_width)` 与 hasSENet）"""
        self.log.info("进行add变异:")
        cardinality_index=np.random.choice(self.cpos)
        group_width_index=np.random.choice(self.gwpos)
        # cardinality*group_width的上限为64*4
        while(self.cardinalitys[cardinality_index]*self.group_widths[group_width_index]>self.cardinality_with_group_width_up_limit or self.cardinalitys[cardinality_index]*self.group_widths[group_width_index]<self.cardinality_with_group_width_down_limit):
            #TODO 这种选择会使得64和32出现的概率变低
            cardinality_index=np.random.choice(self.cpos)
            group_width_index=np.random.choice(self.gwpos)
        indi.units[unit_pos].cardinalitys.insert(block_pos,self.cardinalitys[cardinality_index])
        indi.units[unit_pos].group_widths.insert(block_pos,self.group_widths[group_width_index])
        
        add_senet=np.random.random()>0.5 #大于0.5且现有的senet数量小于最大数量，添加senet
        if add_senet and indi.exist_senet<indi.total_SENet_amount:
            indi.units[unit_pos].hasSENets.insert(block_pos,True)
        else:
            indi.units[unit_pos].hasSENets.insert(block_pos,False)
        self.log.info("    在第{}个unit的{}个位置添加一个block，block参数：cardinality:{}，group_width:{}，hasSENet:{}".format(unit_pos+1,block_pos+1,self.cardinalitys[cardinality_index],self.group_widths[group_width_index],add_senet))
        # 如果添加在最后一个位置，需要修改out_channel
        if block_pos==indi.units[unit_pos].block_amount:
            indi.units[unit_pos].out_channel=ResNeXtBlock.expansion*self.cardinalitys[cardinality_index]*self.group_widths[group_width_index]
            self.log.info("    添加到最后一个位置，修改out_channel:{}".format(indi.units[unit_pos].out_channel))
            # 如果添加的unit位置不是最后一个unit，还需要修改下一个unit的in_channel
            if(unit_pos!=indi.unit_length-1):
                indi.units[unit_pos+1].in_channel=indi.units[unit_pos].out_channel  
                self.log.info("    修改下一个unit的in_channel:{}".format(indi.units[unit_pos+1].in_channel))
        indi.units[unit_pos].block_amount+=1
        


    def remove_block(self,indi,unit_pos,block_pos):
        """删除指定位置的 block，并维护通道一致性"""
        self.log.info("进行remove变异:")
        cardinality=indi.units[unit_pos].cardinalitys.pop(block_pos)
        group_width=indi.units[unit_pos].group_widths.pop(block_pos)
        hasSENet=indi.units[unit_pos].hasSENets.pop(block_pos)
        self.log.info("    删除第{}个unit的{}个位置上的block，block参数：cardinality:{}，group_width:{}，hasSENet:{}".format(unit_pos+1,block_pos+1,cardinality,group_width,hasSENet))
        # 如果删除最后一个位置，需要修改out_channel
        if block_pos==indi.units[unit_pos].block_amount-1:
            indi.units[unit_pos].out_channel=ResNeXtBlock.expansion*indi.units[unit_pos].cardinalitys[-1]*indi.units[unit_pos].group_widths[-1]
            self.log.info("    删除最后一个位置，修改out_channel:{}".format(indi.units[unit_pos].out_channel))
            # 如果删除的unit位置不是最后一个unit，还需要修改下一个unit的in_channel
            if(unit_pos!=indi.unit_length-1):
                indi.units[unit_pos+1].in_channel=indi.units[unit_pos].out_channel
                self.log.info("    修改下一个unit的in_channel:{}".format(indi.units[unit_pos+1].in_channel))
        indi.units[unit_pos].block_amount-=1

    def alter_block(self,indi,unit_pos,block_pos):
        """修改指定位置的 block 参数（合法随机）与 SENet 标记，并维护通道一致性"""
        self.log.info("进行alter变异:")
        cardinality_index=np.random.choice(self.cpos)
        group_width_index=np.random.choice(self.gwpos)
        # cardinality*group_width的上限为64*4
        while(self.cardinalitys[cardinality_index]*self.group_widths[group_width_index]>self.cardinality_with_group_width_up_limit or self.cardinalitys[cardinality_index]*self.group_widths[group_width_index]<self.cardinality_with_group_width_down_limit):
            #TODO 这种选择会使得64和32出现的概率变低
            cardinality_index=np.random.choice(self.cpos)
            group_width_index=np.random.choice(self.gwpos)
        indi.units[unit_pos].cardinalitys[block_pos]=self.cardinalitys[cardinality_index]
        indi.units[unit_pos].group_widths[block_pos]=self.group_widths[group_width_index]
        set_senet=np.random.random()>0.5 #大于0.5且现有的senet数量小于最大数量，添加senet,否则设置为False
        if set_senet:
            if indi.exist_senet<indi.total_SENet_amount:
                indi.units[unit_pos].hasSENets[block_pos]=True
        else:
            indi.units[unit_pos].hasSENets[block_pos]=False
        self.log.info("    修改第{}个unit的{}个位置上的block，block参数：cardinality:{}，group_width:{}，hasSENet:{}".format(unit_pos+1,block_pos+1,self.cardinalitys[cardinality_index],self.group_widths[group_width_index],set_senet))
        # 如果修改最后一个位置，需要修改out_channel
        if block_pos==indi.units[unit_pos].block_amount-1:
            indi.units[unit_pos].out_channel=ResNeXtBlock.expansion*indi.units[unit_pos].cardinalitys[-1]*indi.units[unit_pos].group_widths[-1]
            self.log.info("    修改最后一个位置，修改out_channel:{}".format(indi.units[unit_pos].out_channel))
            # 如果添加的unit位置不是最后一个unit，还需要修改下一个unit的in_channel
            if(unit_pos!=indi.unit_length-1):
                indi.units[unit_pos+1].in_channel=indi.units[unit_pos].out_channel
                self.log.info("    修改下一个unit的in_channel:{}".format(indi.units[unit_pos+1].in_channel))

    def do_mutation(self):
        """对备选 t 个体依次执行一次随机变异

        Returns:
            list[Individual]: 返回生成的子代（原地修改副本）
        """
        for index,offspring in enumerate(self.offsprings):
            self.log.info("个体{}开始变异".format(offspring.id))
            mutation_op=np.random.randint(-1,2) # 变异操作：-1：删除block；0：修改block参数；1：增加block
            # 随机选择变异的block位置
            mutation_unit_pos=np.random.randint(1,offspring.unit_length) 
            mutation_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount)
            # add操作在最后一个block的下一个位置添加，所以+1，remove和alter不需要
            add_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount+1)
            # 增加block
            if mutation_op==1:
                # 如果当前待增加的unit中的block数量已达到上限，则选择其他unit添加，如果均到达上限，则选择其他两种变异操作
                valid=False
                for unit in offspring.units[1:]:
                    if unit.block_amount<offspring.max_ResNeXt_amount:
                        valid=True
                        break
                if valid:
                    while offspring.units[mutation_unit_pos].block_amount==offspring.max_ResNeXt_amount:
                        mutation_unit_pos=self.get_new_mutation_unit_pos(mutation_unit_pos,offspring)
                    add_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount+1)
                    self.add_block(offspring,mutation_unit_pos,add_block_pos)
                else:
                    mutation_op=np.random.randint(-1,1)
                    mutation_unit_pos=np.random.randint(1,offspring.unit_length) 
                    mutation_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount)
                    # 删除block
                    if mutation_op==-1:
                        self.remove_block(offspring,mutation_unit_pos,mutation_block_pos)
                    # 修改block
                    elif mutation_op==0:
                        self.alter_block(offspring,mutation_unit_pos,mutation_block_pos)
            # 删除block
            elif mutation_op==-1:
                # 如果当前待删除的unit中的block数量已达下限，则选择其他unit删除，如果均到达下限，则选择其他两种变异操作
                valid=False
                for unit in offspring.units[1:]:
                    if unit.block_amount>offspring.min_ResNeXt_amount:
                        valid=True
                        break
                if valid:
                    while offspring.units[mutation_unit_pos].block_amount==offspring.min_ResNeXt_amount:
                        mutation_unit_pos=self.get_new_mutation_unit_pos(mutation_unit_pos,offspring)
                    mutation_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount)
                    self.remove_block(offspring,mutation_unit_pos,mutation_block_pos)
                else:
                    mutation_op=np.random.randint(0,2)
                    mutation_unit_pos=np.random.randint(1,offspring.unit_length)
                    mutation_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount)
                    # add操作在最后一个block的下一个位置添加，所以+1，remove和alter不需要
                    add_block_pos=np.random.randint(0,offspring.units[mutation_unit_pos].block_amount+1)
                    # 增加block
                    if mutation_op==1:
                        self.add_block(offspring,mutation_unit_pos,add_block_pos)
                    # 修改block
                    elif mutation_op==0:
                        self.alter_block(offspring,mutation_unit_pos,mutation_block_pos)
            # 修改block
            elif mutation_op==0:
                self.alter_block(offspring,mutation_unit_pos,mutation_block_pos)
            
        return self.offsprings
    
    def get_new_mutation_unit_pos(self,old_unit_pos,indi):
        """重新获取待变异block的unit位置

        Args:
            old_unit_pos (int]): 旧的unit位置
            indi (Individual): 待变异的个体

        Returns:
            [int]: 新的unit位置
        """
        new_pos=np.random.randint(1,indi.unit_length)
        while(old_unit_pos==new_pos):
            new_pos=np.random.randint(1,indi.unit_length)
        return new_pos
"""结构变异操作集合

提供三类原子变异：
- add_block: 在指定 unit 的指定位置插入一个新的 ResNeXt block
- remove_block: 删除一个现有 block
- alter_block: 修改一个现有 block 的 `(cardinality, group_width)` 与 SENet 标记

变异后的通道连贯性
- 若修改/插入/删除的是 unit 的最后一个 block，则需重算该 unit 的 `out_channel`
- 若该 unit 不是最后一个，则还需同步更新下一个 unit 的 `in_channel`
"""
