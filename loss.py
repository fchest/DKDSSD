from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable



class A_softmax(nn.Module):
    def __init__(self, gamma=0):
        super(A_softmax, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte().bool()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss
    
    
def enhance_loss_function(target, ipt, n_frames_list, device):
    """
    Calculate the MSE loss for variable length dataset.

    ipt: [B, F, T]
    target: [B, F, T]
    """
    if target.shape[0] == 1:
            return torch.nn.functional.mse_loss(target, ipt)

    E = 1e-8
    with torch.no_grad():
        masks = []
        for n_frames in n_frames_list:
            masks.append(torch.ones(n_frames, target.size(2), dtype=torch.float32))  # the shape is (T_real, F)

        binary_mask = pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)  # ([T1, F], [T2, F]) => [B, T, F] => [B, F, T]

    
    masked_ipt = ipt.squeeze(1) * binary_mask  # [B, F, T]
    # print("masked_ipt=================", masked_ipt)
    masked_target = target.squeeze(1) * binary_mask # [B, F, T]
    # return ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)  # 不算 pad 部分的贡献，仅计算有效值
    loss = ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)
    # print("enloss--------------- ")
    # print(loss.item())
    return loss