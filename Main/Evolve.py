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

if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

class Evolve():
    def __init__(self):
        self.total_gen=params["gen"]
        self.population=Population(params,1)
        self.log=Log()
        self.evaluateIndi=EvaluateIndi(self.log)
        self.log_path=params["log_path"]
        self.train_top_k=params["train_top_k"]
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
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if os.path.exists(os.path.join(self.log_path,"main.log")):
            os.remove(os.path.join(self.log_path,"main.log"))
        self.log.info("-"*50+"[参数设置]"+"-"*50)
        for k,v in params.items():
            self.log.info("{}:{}".format(k,v))
        self.log.info("{}:{}".format("init_cardinality",params["cardinality"][:params["init_c_and_gw_index"]]))
        self.log.info("{}:{}".format("init_group_width",params["group_width"][:params["init_c_and_gw_index"]]))
        self.log.info("-"*50+"[种群初始化]"+"-"*50+"\n")
        self.population.initialize()
        indis=self.population.individuals
        for indi in indis:
            self.evaluateIndi.evaluate_indi_by_ntk(indi)
            self.log.info(indi)
            for unit in indi.units:
                print(unit)
        self.save_init_indis(indis)
        self.log.info("-"*50+"[种群开始进化]"+"-"*50+"\n")
        # 种群进化
        for gen in range(self.total_gen):
            self.population.cal_indi_spantime()
            self.log.info("*"*30+"当前种群位于第{}代进化".format(gen+1)+"*"*30)
            for indi in self.population.individuals:
                print(indi)
            mutation=Mutation(self.log,indis,gen+1)
            mutation.process()
            self.save_best_indi(indis)
            self.population.gen_id+=1
            if gen!=self.total_gen-1:
                self.population.rename_indi_id()
        self.log.info("-"*50+"[种群进化结束]"+"-"*50+"\n")
        self.log.info("最后一代个体：")
        # 保存最后一代的所有个体
        for indi in self.population.individuals:
            self.log.info(indi)
        self.save_last_gen_indis(self.population.individuals)
        
        # 训练前k个体
        self.log.info("-"*50+"[训练前{}个体]".format(self.train_top_k)+"-"*50+"\n")

        self.train_topk_indis(self.population.individuals)
        
        self.log.info("-"*50+"[个体训练结束]"+"-"*50)

    def train_topk_indis(self,indis):
        """训练ntk前k个个体
        """
        indis.sort(key=lambda indi:indi.ntk)
        k=self.train_top_k
        for indi in indis[:k]:
            self.evaluateIndi.evaluate_indi_by_acc(indi)
            self.save_trained_indi(indi)

    def save_init_indis(self,indis):
        """保存初始化种群
        """
        dir = os.path.join(self.log_path,"init_indis")
        if not os.path.exists(dir):
                os.makedirs(dir)
        for indi in indis:
            with open(os.path.join(dir,indi.id+ "_"+str(indi.ntk) + '.txt'), 'wb') as file:
                pickle.dump(indi, file) 

    def save_last_gen_indis(self,indis):
        """保存最后一代的所有个体
        """
        dir = os.path.join(self.log_path,"last_gen_indis")
        if not os.path.exists(dir):
                os.makedirs(dir)
        for indi in indis:
            with open(os.path.join(dir,indi.id+ "_"+str(indi.ntk) + '.txt'), 'wb') as file:
                pickle.dump(indi, file) 

    def save_trained_indi(self,indi):
        """保存已训练的个体
        """
        dir = os.path.join(self.log_path,"trained_indis")
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir,indi.id+ "_"+str(indi.ntk) +"_"+str(indi.acc) + '.txt'), 'wb') as file:
            pickle.dump(indi, file)

    def save_best_indi(self,indis):
        """
        基于ntk保存每代的最优个体
        """
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
