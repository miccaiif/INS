import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
from alexnet import teacher_Attention_head
class INS(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.dataset == 'cub200' ## False in cifar-10
        # we allow pretraining for CUB200, or the network will not converge

        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

        # init a bag prediction head: FC layer
        self.bag_pred_head = None

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.model_teacherHead = teacher_Attention_head()
        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))  ## (8192,128)
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class))  ## (8192,10)  伪标签
        self.register_buffer("queue_partial", torch.randn(args.moco_queue, args.num_class))  ## (8192,10) 偏标签
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))  ## 0(10,128)
        self.bag_classifier = nn.Linear(512, 2)
    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, partial_Y, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        partial_Y = concat_all_gather(partial_Y)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # print('val:',args.moco_queue, batch_size)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size, :] = labels
        self.queue_partial[ptr:ptr + batch_size, :] = partial_Y
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y, p_y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        p_y_gather = concat_all_gather(p_y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, p_y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        p_y_gather = concat_all_gather(p_y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this]

    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes
        
    def bag_head(self, features):
        # mean-pooling attention-pooling
        bag_prediction, _, _, _ = self.model_teacherHead(features, returnBeforeSoftMaxA=True, scores_replaceAS=None)
        # print('bag_prediction:',bag_prediction)
        return bag_prediction
        # return features.mean(dim=0, keepdim=True)
    
    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False, bag_flag=False):
        if bag_flag:
            # print('img_q.shape:',img_q.shape) # 100*3*32*32
            # code for cal bag predictions, input is bag ()
            # features = encoder(all instance from a bag)
            features = self.encoder_q.encoder(img_q) # 100*512
            bag_prediction = self.bag_head(features)   # 512
        
            # or Max-pooling, Attention-pooling
            # bag_prediction = self.bag_classifier(bag_feature)
            # print(bag_prediction.shape, bag_prediction)
            return bag_prediction
            # return 0
        else:
            # 先提取q的特征，然后再提取im_k的特征时先声明不进行梯度更新，先进行momentum更新关键字网络，
            # 然后对于im_k的索引进行shuffle，然后再提取特征，提取完特征之后又恢复了特征k的顺序（unshuffle_ddp），
            # 因为q的顺序没有打乱，在计算损失时需要对应。

            # output是classifier的输出结果，q是MLP的输出结果
            # 两个样本的预测标签相同，则他们为正样本对，反之则为负样本对
            # if point == 0: 相当于是个mask ，每次都要对结果mask
            output, q = self.encoder_q(img_q)  ##([256,10]),([256,128])
            if eval_only:
                return output, q
            # for testing
            # 分类器的预测结果：torch.softmax(output, dim=1)
            # 所以病理的图片标签应该是 pos,neg 都为1
            predicted_scores = torch.softmax(output, dim=1) * partial_Y # 分类器结果，限制标签在候选集y当中
            max_scores, pseudo_labels = torch.max(predicted_scores, dim=1) ## values, index（预测标签）
            # using partial labels to filter out negative labels

            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()  ## 0([10,128])
            logits_prot = torch.mm(q, prototypes.t())  ##([256,10])
            score_prot = torch.softmax(logits_prot, dim=1) ##([256,10]) # 成绩是平均的

            # update momentum prototypes with pseudo labels
            for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels)): #concat_all_gather将多卡数据合并
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat  ## 按feature_c更新prototypes
                # torch.set_printoptions(profile="full")
                # print((1-args.proto_m)*feat)
            # normalize prototypes
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)  ##(10,128)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder(args)  # update the momentum encoder with encoder_q
                # shuffle for making use of BN
                im_k, predicted_scores, partial_Y, idx_unshuffle = self._batch_shuffle_ddp(im_k, predicted_scores, partial_Y)
                _, k = self.encoder_k(im_k)  ## 输出k样本预测类别
                # print('img_k',im_k.shape,k.shape)
                # undo shuffle
                k, predicted_scores, partial_Y = self._batch_unshuffle_ddp(k, predicted_scores, partial_Y, idx_unshuffle)

            features = torch.cat((q, k, self.queue.clone().detach()), dim=0)  ##
            pseudo_scores = torch.cat((predicted_scores, predicted_scores, self.queue_pseudo.clone().detach()), dim=0)
            partial_target = torch.cat((partial_Y, partial_Y, self.queue_partial.clone().detach()), dim=0)
            # to calculate SupCon Loss using pseudo_labels and partial target

            # dequeue and enqueue
            self._dequeue_and_enqueue(k, predicted_scores, partial_Y, args)
            # 分类器预测结果，伪标签，原型的结果
            return output, features, pseudo_scores, partial_target, score_prot


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

