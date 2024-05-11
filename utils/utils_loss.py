import torch
import torch.nn.functional as F
import torch.nn as nn


class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.8, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)


    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        # 分类器输出的结果和伪标签的乘积
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        #
        # _, target = (self.confidence[index, :]).max(dim=1)
        # logp = self.ce(outputs, target)
        # p = torch.exp(-logp)
        # loss = (1-p)**self.gamma * logp
        # return loss
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY, point = None):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            # 对confidence进行更新
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                              + (1 - self.conf_ema_m) * pseudo_label
            teacher_label = []
            pred_score = []
            for i in range(point.shape[0]):
                # print(labels[i])
                if (point[i].equal(torch.tensor(0))):
                    self.confidence[batch_index[i],:] = torch.tensor([1,0])



class Cls_loss(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.confidence = labels
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        # 分类器输出的结果和伪标签的乘积
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        _, i = (self.confidence[index,:]).max(dim=1)

        return average_loss

class Focalloss(nn.Module):
    def __init__(self, labels, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(Focalloss,self).__init__()
        self.confidence = labels
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, outputs, index):
        _, target = (self.confidence[index, :]).max(dim=1)
        logp = self.ce(outputs, target)
        p = torch.exp(-logp)
        loss = (1-p)**self.gamma * logp
        return loss
        # return average_loss


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss