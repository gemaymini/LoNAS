import torch as t
import torch.nn as nn
import torch.optim as optim
import sys
import os
import shutil
import time
import gc
import torch.backends.cudnn as cudnn
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
dir=os.path.abspath("..")
sys.path.append(dir)
np.random.seed(0)
cudnn.benchmark = True
t.manual_seed(0)
cudnn.enabled=True
t.cuda.manual_seed(0)


from newEvaluate.ConstructNet import ConstructNet
from newEvaluate.EvolveDataLoader_imagenet import Evolve_DataLoader
from newSetting.config import Config
from Tool.utils import cal_indi_parameters_and_flops, progress_bar


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
# 训练网络
class TrainNet():

    def __init__(self,params,log):
        self.params=params
        self.dataset=params["dataset"]
        self.lr=params["lr"]
        self.device=params["device"]
        # self.mutilpgu=params["gpu"]
        self.epochs=params["evolve_epochs"]
        self.log=log
        self.log_path=params["log_path"]
        self.es=0
        self.num_classes=1000


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
        """初始化 Conv/BN/Linear 权重（Xavier/常数/正态）"""
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
    def accuracy(self,output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with t.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        t.save(state, filename)  
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
    
    def train(self,individual,train_loader=None,test_loader=None):
        """执行训练与周期性验证，并将最佳指标写入日志与个体"""
        data_loader=Evolve_DataLoader(self.dataset,False)
        train_loader,test_loader=data_loader.data_loader()
        net=ConstructNet(individual,self.num_classes)
        self.initialize_weight(net)
        net = nn.DataParallel(net)
        net =net.cuda()
        # print(net)
        criterion=nn.CrossEntropyLoss().cuda()
        optimizer=optim.SGD(net.parameters(),lr=self.lr,momentum=0.9,weight_decay=5e-4)
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        best_top1,best_top5=0,0
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log.info("个体{}开始训练".format(individual.id))
        with open(os.path.join(self.log_path,individual.id+"_test_acc.log"),"w") as test_f:
            with open(os.path.join(self.log_path,individual.id+"_train_acc.log"),"w") as train_f:
                for epoch in range(self.epochs):
                    batch_time = AverageMeter('Time', ':6.3f')
                    data_time = AverageMeter('Data', ':6.3f')
                    losses = AverageMeter('Loss', ':.4e')
                    top1 = AverageMeter('Acc@1', ':6.2f')
                    top5 = AverageMeter('Acc@5', ':6.2f')
                    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
                    # self.adjust_learning_rate(optimizer, epoch,)
                    print('\nEpoch: %d' % epoch)
                    net.train()
                    end = time.time()
                    
                    for index,data in enumerate(train_loader): 
                        # if index==16:break 
                        data_time.update(time.time() - end)
                        length=len(train_loader)
                        inputs,labels=data
                        inputs=inputs.cuda(non_blocking=True)
                        labels=labels.cuda(non_blocking=True)
                        cudnn.enabled=True
                        # inputs=inputs.cuda()
                        # labels=labels.cuda()
                        optimizer.zero_grad()
                        outputs=net(inputs)
                        loss=criterion(outputs,labels)
                        acc1, acc5 = self.accuracy(outputs, labels, topk=(1, 5))
                        losses.update(loss.item(), inputs.size(0))
                        top1.update(acc1[0], inputs.size(0))
                        top5.update(acc5[0], inputs.size(0))
                        loss.backward()
                        optimizer.step()
                        del loss
                        t.cuda.empty_cache()
                        gc.collect()
                        batch_time.update(time.time() - end)
                        end = time.time()
                        if index % 10 == 0:
                            progress.print(index)
                    if (epoch+1)%2==0:
                        batch_time = AverageMeter('Time', ':6.3f')
                        losses = AverageMeter('Loss', ':.4e')
                        top1 = AverageMeter('Acc@1', ':6.2f')
                        top5 = AverageMeter('Acc@5', ':6.2f')
                        with t.no_grad():
                            end = time.time()
                            net.eval()
                            for index,data in enumerate(test_loader):
                                # if index==16:break
                                inputs,labels=data
                                inputs=inputs.cuda()
                                labels=labels.cuda()
                                # inputs=inputs.cuda()
                                # labels=labels.cuda()
                                outputs=net(inputs)
                                loss=criterion(outputs,labels).item()
                                acc1, acc5 = self.accuracy(outputs, labels, topk=(1, 5))
                                losses.update(loss.item(), inputs.size(0))
                                top1.update(acc1[0], inputs.size(0))
                                top5.update(acc5[0], inputs.size(0))
                            test_f.write("EPOCH=%03d,Acc1= %.3f%%,Acc5= %.3f%%\n" % (epoch + 1, top1.avg,top5.avg))
                            if top1.avg>best_top1:
                                best_top1=top1.avg
                                self.es=0
                            if top5.avg>best_top5:
                                best_top5=top5.avg
                            # else:
                            #     if epoch >290:
                            #         self.es+=1
                            #         print("Counter {} of 8".format(self.es))
                            #         if self.es>7:
                            #             self.log.info("Early Stopping with best_acc:{} and val_acc for this epoch:{}".format(best_acc,epoch))
                            #             break
                        
                    scheduler.step()
                    # self.adjust_lr(optimizer,epoch,self.lr)
        self.log.info("v0215 best_acc1:{} and best_acc5:{} with re".format(best_top1,best_top5))
        individual.acc=best_top1

if __name__=="__main__":
   
    from newGA.Individual import Individual
    from Tool.Log import Log
    import pickle
    params=Config.pro_params
    print(params)
    path="../Output/trained_indis"
    indis=os.listdir(path)
    indis.sort()
    log=Log()
    log.info("")
    log.info("another 10 runs")
    for _ in range(20):
        with open(os.path.join(path,"test.txt"),"rb") as file:
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
"""ImageNet 训练器

职责
- 构建模型与优化器，执行若干 epoch 的训练与验证
- 打印 epoch 级/批次级统计，记录最佳 Top-1/Top-5
"""
