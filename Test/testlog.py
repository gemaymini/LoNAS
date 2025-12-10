import sys
import os
dir=os.path.abspath('..')
sys.path.append(dir)

from Search.Evolve import Evolve

if __name__=="__main__":
    ev=Evolve(3)
    ev.initialize()
    ev.do_evolve()
"""日志与进化流程的简单调用脚本（示例）"""
