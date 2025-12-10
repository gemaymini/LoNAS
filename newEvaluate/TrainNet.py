import torch as t
import torch.nn as nn
import torch.optim as optim
import sys
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import numpy as np
dir=os.path.abspath("..")
sys.path.append(dir)



from newEvaluate.ConstructNet import ConstructNet
from newEvaluate.EvolveDataLoader_re import Evolve_DataLoader
from newSetting.config import Config
from Tool.utils import cal_indi_parameters_and_flops, progress_bar


# 训练网络
class TrainNet():

    def __init__(self,params,log):
        self.params=params
        self.dataset=params["dataset"]
        self.lr=params["lr"]
        self.device=params["device"]
        self.mutilpgu=params["gpu"]
        self.epochs=params["evolve_epochs"]
        self.log=log
        self.log_path=params["log_path"]
        self.es=0
        self.num_classes=10 if params["dataset"]==1 else 100


    def adjust_lr(self,optim,epoch,lr):
        lr=lr*(0.8**(epoch//10))
        for param_group in optim.param_groups:
            param_group["lr"]=lr
    def adjust_learning_rate(self,optimizer, epoch):
        if epoch in [150,225]:
            self.lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
    
    def initialize_weight(self,net):
        """初始化网络权重（Conv/BN/Linear）"""
        for m in net.modules():
            if isinstance(m,nn.Conv2d):
                t.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                t.nn.init.normal_(m.weight.data,0,0.01)
                m.bias.data.zero_()
    
    def train(self,individual):
        """执行 CIFAR 训练与周期性验证，并写回最佳精度"""
        data_loader=Evolve_DataLoader(self.dataset,False)
        train_loader,test_loader=data_loader.data_loader()
        net=ConstructNet(individual,self.num_classes)
        self.initialize_weight(net)
        if self.mutilpgu!="None":
            # device_ids = [0, 1]
            net = nn.DataParallel(net, device_ids=[0,1])
        net.to(self.device)
        # print(net)
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=self.lr,momentum=0.9,weight_decay=5e-4)
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        best_acc=0
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log.info("个体{}开始训练".format(individual.id))
        with open(os.path.join(self.log_path,individual.id+"_test_acc.log"),"w") as test_f:
            with open(os.path.join(self.log_path,individual.id+"_train_acc.log"),"w") as train_f:
                for epoch in range(self.epochs):
                    # self.adjust_learning_rate(optimizer, epoch,)
                    print('\nEpoch: %d' % epoch)
                    net.train()
                    sum_loss=0.0
                    correct=0.0
                    total=0.0
                    j=0
                    for index,data in enumerate(train_loader):
                        length=len(train_loader)
                        inputs,labels=data
                        inputs=inputs.to(self.device)
                        labels=labels.to(self.device)
                        # inputs=inputs.cuda()
                        # labels=labels.cuda()
                        optimizer.zero_grad()
                        outputs=net(inputs)
                        loss=criterion(outputs,labels)
                        loss.backward()
                        optimizer.step()
                        sum_loss+=loss.item()
                        j+=1
                        _,preds=t.max(outputs.data,1)
                        total+=labels.size(0)
                        correct+=preds.eq(labels.data).sum().item()
                        progress_bar(index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (sum_loss/(index+1), 100.*correct/total, correct, total))
                    train_f.write('%03d |Loss: %.03f | Acc: %.3f%%\n'
                            % (epoch + 1, sum_loss /j,
                            100. * float(correct) / total))

                    if (epoch+1)%2==0:
                        with t.no_grad():
                            correct=0
                            total=0
                            sum_loss=0
                            net.eval()
                            for index,data in enumerate(test_loader):
                                inputs,labels=data
                                inputs=inputs.to(self.device)
                                labels=labels.to(self.device)
                                # inputs=inputs.cuda()
                                # labels=labels.cuda()
                                outputs=net(inputs)
                                sum_loss+=criterion(outputs,labels).item()
                                _, preds = t.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (preds == labels).sum().item()
                                progress_bar(index, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (sum_loss/(index+1), 100.*correct/total, correct, total))
                            acc = 100. * float(correct) / total
                            test_f.write("EPOCH=%03d,Accuracy= %.3f%%\n" % (epoch + 1, acc))
                            if acc>best_acc:
                                best_acc=acc
                                self.es=0
                            # else:
                            #     if epoch >290:
                            #         self.es+=1
                            #         print("Counter {} of 8".format(self.es))
                            #         if self.es>7:
                            #             self.log.info("Early Stopping with best_acc:{} and val_acc for this epoch:{}".format(best_acc,epoch))
                            #             break
                        
                    scheduler.step()
                    # self.adjust_lr(optimizer,epoch,self.lr)
        self.log.info("v429 best_acc:{} with re".format(best_acc))
        individual.acc=best_acc

if __name__=="__main__":
    from newGA.Individual import Individual
    from Tool.Log import Log
    import pickle
    params=Config.dev_params
    print(params)
    path="../trained_indis"
    indis=os.listdir(path)
    indis.sort()
    log=Log()
    log.info("")
    log.info("another 10 runs")
    for _ in range(20):
        with open(os.path.join(path,"6001_3.548_95.69.txt"),"rb") as file:
            indi=pickle.load(file)
        # cal_indi_parameters_and_flops(indi)
        print(indi)
        log.info(indi)
        # for unit in indi.units[1:]:
            # l=unit.block_amount
            # r=np.random.randint(0,l)
            # for i in range(l):
            #     unit.hasSENets[i]=True if np.random.randint(0,3)>1 else False
            # print(unit)
        cal_indi_parameters_and_flops(indi)
        print(indi)
        trainnet=TrainNet(params,log)
        trainnet.train(indi)    
"""CIFAR 训练器（RandomErasing 版本的数据管道）"""
