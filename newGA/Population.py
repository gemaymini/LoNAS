import sys
import os

dir=os.path.abspath("..")
sys.path.append(dir)

from newGA.Individual import Individual
from newSetting.config import Config
import numpy as np
import hashlib
import copy

class Population():
    def __init__(self,params,gen_id):
        """初始化种群元信息"""
        self.gen_id=gen_id #种群当前经历的进化代次
        self.indi_id=1 # 种群中最后加入的个体的编号
        self.pop_size=params["pop_size"]
        self.params=params
        self.individuals=[] # 种群中的个体列表

    def initialize(self):
        """按 `pop_size` 生成个体并赋予两位代次+两位序号的 id"""
        for _ in range(self.pop_size):
            ind_id="%02d%02d"%(self.gen_id,self.indi_id)
            indi=Individual(self.params,ind_id)
            indi.initialize()
            self.individuals.append(indi)
            self.indi_id+=1
    
    def add_offsprings_to_parents(self,offsprings):
        """将生成的子代加入到父代组成新的种群

        Args:
            offsprings: 新生成的子代列表
        """
        for offspring in offsprings:
            indi=copy.deepcopy(offspring)
            indi_id="%02d%02d"%(self.gen_id,self.indi_id)
            indi.id=indi_id
            self.individuals.append(indi)
            self.indi_id+=1
        self.pop_size+=len(offsprings)
    
    def rename_indi_id(self):
        """在一代结束后重新对个体进行顺序编号（保持两位序号）"""
        self.indi_id=1
        for indi in self.individuals:
            indi.id="%02d%02d"%(self.gen_id,self.indi_id)
            self.indi_id+=1

    def cal_indi_spantime(self):
        """更新个体寿命计数：每经历一代 +1"""
        for indi in self.individuals:
            indi.spantime+=1
        
if __name__=="__main__":
   pass
"""种群管理

职责
- 初始化一代种群、维护个体编号与寿命统计
- 在进化过程中合并子代、重编号、更新大小
"""
