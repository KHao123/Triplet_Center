import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pesTriplet(embeddings,labels,centers,lambd):
    C = centers.size()[0]
    n = embeddings.size()[0]
    labelSet = torch.arange(C)
    pes_triplet = []
    for i in range(n):
        embedding = embeddings[i]
        label = labels[i]
        dis_to_centers = (embedding.repeat((C,1)) - centers).pow(2).sum(1)
        ap_dis = dis_to_centers[label]
        loss_val = lambd + ap_dis - dis_to_centers
        #print(loss_val)
        mask = loss_val.gt(0)
        mask[label] = 0
        if torch.sum(mask) > 0:
            neg = labelSet[mask]
            pes_triplet += [[i,label,ex_label] for ex_label in neg]
    if len(pes_triplet)==0:
        return None
    return torch.LongTensor(pes_triplet).cuda() if embeddings.is_cuda else torch.LongTensor(pes_triplet)

def get_minTriplet(embeddings,labels,centers,lambd):
    C = centers.size()[0]
    n = embeddings.size()[0]
    labelSet = torch.arange(C)
    pes_triplet = []
    for i in range(n):
        embedding = embeddings[i]
        label = labels[i]
        dis_to_centers = (embedding.repeat((C,1)) - centers).pow(2).sum(1)
        ap_dis = dis_to_centers[label]
        loss_val = lambd + ap_dis - dis_to_centers
        #print(loss_val)
        argmin = torch.argmax(loss_val)
        
        if argmin!=label and loss_val[argmin]>0:
            pes_triplet += [[i,label,argmin]]
    if len(pes_triplet)==0:
        return None
    return torch.LongTensor(pes_triplet).cuda() if embeddings.is_cuda else torch.LongTensor(pes_triplet)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineCenterLoss(nn.Module):
    def __init__(self,lambd):
        super(OnlineCenterLoss,self).__init__()
        self.lambd = lambd

    def forward(self, embeddings, targets, centers):
        triplets = get_pesTriplet(embeddings,targets,centers,self.lambd) # A:embedding P:center N:center(false)
        if triplets is None:
            zero = torch.Tensor([0.])
            zero.requires_grad_()
            return zero
        #print(triplets)
        ap_distances = (embeddings[triplets[:,0]] - centers[triplets[:,1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:,0]] - centers[triplets[:,2]]).pow(2).sum(1)
        
        losses = F.relu(ap_distances - an_distances + self.lambd)
        return losses.mean()

class OnlineCenterLossV2(nn.Module):
    def __init__(self,lambd):
        super(OnlineCenterLossV2,self).__init__()
        self.lambd = lambd

    def forward(self, embeddings, targets, centers):
        triplets = get_minTriplet(embeddings,targets,centers,self.lambd) # A:embedding P:center N:center(false)
        if triplets is None:
            zero = torch.Tensor([0.])
            zero.requires_grad_()
            return zero
        #print(triplets)
        ap_distances = (embeddings[triplets[:,0]] - centers[triplets[:,1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:,0]] - centers[triplets[:,2]]).pow(2).sum(1)
        
        losses = F.relu(ap_distances - an_distances + self.lambd)
        return losses.mean()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class LiftedEmbeddingLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(LiftedEmbeddingLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        losses = 0
        for i in range(len(triplets)):
            triplet = triplets[i]
            ap_distances = (embeddings[triplet[0]] - embeddings[triplet[1]]).pow(2).sum(1)  # .pow(.5)
            an_distances = (embeddings[triplet[0]] - embeddings[triplet[2]]).pow(2).sum(1)  # .pow(.5)

            ap_exp_sum = torch.exp(ap_distances).sum()
            an_exp_sum = torch.exp(self.margin - an_distances).sum()

            ap = torch.log(ap_exp_sum)
            an = torch.log(an_exp_sum)

            losses += F.relu(ap+an)

        return losses, len(triplets)