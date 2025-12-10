import sys
import os
dir=os.path.abspath("..")
sys.path.append(dir)
import pickle
from newEvaluate.EvaluateIndi import EvaluateIndi
from Tool.Log import Log
from Tool.utils import cal_indi_parameters_and_flops
def check_trained_indi():
    path=os.path.join("../Output/v5.15_1/trained_indis")
    indis=os.listdir(path)
    indis.sort()
    for indi_name in indis:
        with open(os.path.join(path,indi_name), 'rb') as file1:
            indi = pickle.load(file1)
            print(indi)
            for unit in indi.units:
                print(unit)
            print()

def retrain_top_k_by_epoch_300():
    #TODO 增加epoch使indi达到收敛
    evaluateIndi=EvaluateIndi(Log())
    path=os.path.join("../Output/v5.20/trained_indis")
    indis=os.listdir(path)
    for indi_name in indis:
        with open(os.path.join(path,indi_name), 'rb') as file1:
            indi = pickle.load(file1)
            cal_indi_parameters_and_flops(indi)
            print(indi)



if __name__=="__main__":
    retrain_top_k_by_epoch_300()