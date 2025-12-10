import numpy as np
import torch

import sys
import os
dir=os.path.abspath("..")
sys.path.append(dir)
from newSetting.config import Config
if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network

# NOTE 获得网络的TNK得分
def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
    #TODO 单GPU
    # device = params["device"]
    device=0
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(xloader):
        # 只做一个batch_size
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            # inputs_ = inputs.cuda(device=0)
            # print(inputs.shape)
            logit = network(inputs)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
            # del inputs
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds
