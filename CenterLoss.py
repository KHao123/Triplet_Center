import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_centers(embeddings,targets,n_classes):
    centers = torch.Tensor([]).cuda()
    for lbl in range(n_classes):
        mask = targets.eq(lbl)
        embeddings_ = embeddings[mask]
        center = embeddings_.mean(dim=0)
        centers = torch.cat([centers,center.unsqueeze(dim=0)])
    assert centers.shape == (n_classes,embeddings.size()[1])
    return centers

def diversity_regularizer(centers,n_classes):
    c_j = torch.Tensor([]).cuda()
    c_k = torch.Tensor([]).cuda()# j < k
    labelSet = torch.arange(n_classes).cuda()
    for lbl in range(n_classes):
        mask = labelSet.eq(lbl)
        gt_mask = labelSet.gt(lbl)
        repeat_n = torch.sum(gt_mask)
        if repeat_n > 0:
            c_j = torch.cat([c_j,centers[mask].repeat(repeat_n,1)])
        c_k = torch.cat([c_k,centers[gt_mask]])

    assert c_j.size() == c_k.size()
    mu = (c_j - c_k).pow(2).sum(1).mean()
    R_w = ((c_j - c_k).pow(2).sum(1) - mu).pow(2).mean()

    return R_w

class CenterLoss(nn.Module):
    def __init__(self,lambd,n_classes):
        super(CenterLoss,self).__init__()
        self.lambd = lambd
        self.n_classes = n_classes

    def forward(self,embeddings,targets,centers):
        repeat_n = self.n_classes - 1
        labelSet = torch.arange(self.n_classes).cuda()

        center_mat = torch.Tensor([]).cuda()
        exc_center_mat = torch.Tensor([]).cuda()

        data_mat = torch.Tensor(embeddings.cpu().data.numpy().repeat(repeat_n,axis=0)).cuda()
        for i in range(embeddings.size()[0]):
            lbl = targets[i]
            exc_center_mask = labelSet.ne(lbl)
            center_mask = labelSet.eq(lbl)
            center_mat = torch.cat([center_mat,centers[center_mask].repeat(repeat_n,1)])
            exc_center_mat = torch.cat([exc_center_mat,centers[exc_center_mask]])

        #print('data:{},center:{},excenter:{}'.format(data_mat.size(),center_mat.size(),exc_center_mat.size()))
        assert center_mat.size() == exc_center_mat.size()
        assert center_mat.size() == data_mat.size()

        dis_intra = (data_mat - center_mat).pow(2).sum(1) 
        dis_inter = (data_mat - exc_center_mat).pow(2).sum(1)
        L_mm = F.relu(self.lambd + dis_intra - dis_inter).mean()
        #R_w = diversity_regularizer(centers,self.n_classes)
        loss = L_mm 

        return loss



