"""LoNAS 主进化脚本

职责
- 按配置初始化种群，基于 NTK 作为适应度进行多代进化（含变异与淘汰）
- 记录每代最优个体、最后一代个体，并在结束后训练 NTK 最优的前 k 个体

关键流程
1) 种群初始化 → 逐个体评估 NTK → 保存初始化快照
2) 循环 gen 次：选择-变异-合并-重编号-淘汰 → 保存当代最优
3) 输出最后一代 → 训练前 k 个体并保存训练结果

注意
- 日志与输出目录由配置提供；运行前会清理历史 best/last/trained 目录
- 仅在结构变更后重新计算统计量与 NTK，避免不必要开销
"""

import sys
import os
dir=os.path.abspath("..")
sys.path.append(dir)

from newGA_OP.doMutation import doMutation
from newGA_OP.Mutation import Mutation
from newSetting.config import Config
from procedures import get_ntk_n
from newEvaluate.ConstructNet import ConstructNet
from newGA.Population import Population
from newGA.Individual import Individual
from newEvaluate.EvaluateIndi import EvaluateIndi
from Tool.Log import Log
import numpy as np
import copy
import pickle
import shutil

# 根据运行环境选择参数集
if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

class Evolve():
    """进化流程控制类

    属性
    - total_gen: 计划进化的代数
    - population: 当前种群对象
    - log: 统一日志器（文件 + 控制台）
    - evaluateIndi: 个体评估器（NTK/ACC）
    - log_path: 输出路径（含日志与快照）
    - train_top_k: 结束后训练的前 k 个体数量
    """
    def __init__(self):
        self.total_gen=params["gen"]
        self.population=Population(params,1)
        self.log=Log()
        self.evaluateIndi=EvaluateIndi(self.log)
        self.log_path=params["log_path"]
        self.train_top_k=params["train_top_k"]
        # 清理旧输出，避免混淆本次结果
        dir = os.path.join(self.log_path,"trained_indis")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        dir = os.path.join(self.log_path,"best_indis")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        dir = os.path.join(self.log_path,"last_gen_indis")
        if os.path.exists(dir):
            shutil.rmtree(dir)

    def do_evolve(self):
        """执行完整进化与训练流程"""
        # 准备日志目录与文件
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if os.path.exists(os.path.join(self.log_path,"main.log")):
            os.remove(os.path.join(self.log_path,"main.log"))
        # 输出参数快照，便于复现与对比
        self.log.info("-"*50+"[参数设置]"+"-"*50)
        for k,v in params.items():
            self.log.info("{}:{}".format(k,v))
        self.log.info("{}:{}".format("init_cardinality",params["cardinality"][:params["init_c_and_gw_index"]]))
        self.log.info("{}:{}".format("init_group_width",params["group_width"][:params["init_c_and_gw_index"]]))

        # 1) 初始化与首轮 NTK 评估
        self.log.info("-"*50+"[种群初始化]"+"-"*50+"\n")
        self.population.initialize()
        indis=self.population.individuals
        for indi in indis:
            self.evaluateIndi.evaluate_indi_by_ntk(indi)
            self.log.info(indi)
            for unit in indi.units:
                print(unit)
        self.save_init_indis(indis)

        # 2) 代际进化
        self.log.info("-"*50+"[种群开始进化]"+"-"*50+"\n")
        for gen in range(self.total_gen):
            # 累积寿命，用于中期的多样性保留策略
            self.population.cal_indi_spantime()
            self.log.info("*"*30+"当前种群位于第{}代进化".format(gen+1)+"*"*30)
            for indi in self.population.individuals:
                print(indi)
            # 选择 t 个体 → 变异 → 合并 → 重编号 → 淘汰
            mutation=Mutation(self.log,indis,gen+1)
            mutation.process()
            self.save_best_indi(indis)
            self.population.gen_id+=1
            if gen!=self.total_gen-1:
                self.population.rename_indi_id()

        # 3) 输出末代与训练 Top-K
        self.log.info("-"*50+"[种群进化结束]"+"-"*50+"\n")
        self.log.info("最后一代个体：")
        for indi in self.population.individuals:
            self.log.info(indi)
        self.save_last_gen_indis(self.population.individuals)

        self.log.info("-"*50+"[训练前{}个体]".format(self.train_top_k)+"-"*50+"\n")
        self.train_topk_indis(self.population.individuals)
        self.log.info("-"*50+"[个体训练结束]"+"-"*50)

    def train_topk_indis(self,indis):
        """训练 NTK 最优的前 k 个体，并保存其训练结果"""
        indis.sort(key=lambda indi:indi.ntk)
        k=self.train_top_k
        for indi in indis[:k]:
            self.evaluateIndi.evaluate_indi_by_acc(indi)
            self.save_trained_indi(indi)

    def save_init_indis(self,indis):
        """保存初始化种群到 `init_indis/`（包含 NTK 快照）"""
        dir = os.path.join(self.log_path,"init_indis")
        if not os.path.exists(dir):
                os.makedirs(dir)
        for indi in indis:
            with open(os.path.join(dir,indi.id+ "_"+str(indi.ntk) + '.txt'), 'wb') as file:
                pickle.dump(indi, file) 

    def save_last_gen_indis(self,indis):
        """保存最后一代的所有个体到 `last_gen_indis/`（包含 NTK 快照）"""
        dir = os.path.join(self.log_path,"last_gen_indis")
        if not os.path.exists(dir):
                os.makedirs(dir)
        for indi in indis:
            with open(os.path.join(dir,indi.id+ "_"+str(indi.ntk) + '.txt'), 'wb') as file:
                pickle.dump(indi, file) 

    def save_trained_indi(self,indi):
        """保存已训练的个体到 `trained_indis/`（文件名包含 NTK 与 ACC）"""
        dir = os.path.join(self.log_path,"trained_indis")
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir,indi.id+ "_"+str(indi.ntk) +"_"+str(indi.acc) + '.txt'), 'wb') as file:
            pickle.dump(indi, file)

    def save_best_indi(self,indis):
        """按 NTK 最小值保存本代最优个体到 `best_indis/`"""
        best_indi=indis[0]
        for indi in indis[1:]:
            if indi.ntk<best_indi.ntk:
                best_indi=indi
        dir = os.path.join(self.log_path,"best_indis")
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir,best_indi.id+ "_"+str(best_indi.ntk) + '.txt'), 'wb') as file:
            pickle.dump(best_indi, file)

    
if __name__=="__main__":
    print("ss")
    evolve=Evolve()
    evolve.do_evolve()
