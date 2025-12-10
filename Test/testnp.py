import numpy as np

if __name__=="__main__":
    l=[False for _ in range(10)]
    index=list(range(10))
    print(np.random.choice(index))
    l=[1]*5+[2]*10
    print(l)
    res=0
    for _ in range(100):
        num=np.random.choice(l)
        if(num==2):
            res+=1
    print(res/100)
    print(10/15) 
    # l=list(range(10))
    # print(l)
    # l.pop(1)
    # l.insert(len(l),100)
    # print(l)
    # a=[False,False,True,True]
    # print(sum(a))
"""Numpy 随机选择与概率分布的简单测试脚本"""
