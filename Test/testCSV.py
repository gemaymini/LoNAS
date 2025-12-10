import sys
import os
dir=os.path.abspath('..')
sys.path.append(dir)
import pickle

def generateCSV():
    path="../Output/v5.23/"
    indis_name=os.listdir(path+"trained_indis")
    indis_name.sort()
    indis=[]
    with open(os.path.join(path,"trained.csv"),"w") as f:
        for indi_name in indis_name:
            with open(os.path.join(path+"trained_indis",indi_name),"rb") as file:
                indi=pickle.load(file)
                indis.append(indi)
        indis.sort(key=lambda indi:indi.ntk)
        for indi in indis:    
            print(indi)
            f.write("{},{},{},{},{},{},{},{},{}\n".format(indi.id,indi.unit_length,indi.conv_length,indi.exist_senet,indi.spantime,indi.ntk,indi.acc,indi.parameters,indi.flops//1000))

if __name__=="__main__":
    generateCSV()
"""训练结果汇总为 CSV 的脚本

说明
- 从 `trained_indis/` 读取已训练个体，按 NTK 排序，输出关键信息到 CSV
"""
