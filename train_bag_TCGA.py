import argparse
import builtins
import os
import random
from sklearn import metrics
import shutil
import time
import warnings
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau

import tensorboard_logger as tb_logger
from model import INS
from resnet import *
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss, Cls_loss
# from utils.cifar_mil_new2 import load_cifarmil
from dataset_CAMELYON16_BasedOnFeat import load_cam16_mil
# from dataset_CervicalCancer_feat import load_Trans
# from dataloader_DScam_based_on_npy import load_TCGA
from dataset_TCGA_LungCancer_return_bag import load_TCGA

# SEED = 0
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper INS')
parser.add_argument('--dataset', default='tcga', type=str,
                    choices=['cifarmil','cam16_mil','trans','tcga'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment_final/TCGA', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 used in INS)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=45, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:50001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="0", type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num-class', default=2, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.9, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95, 0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=20, type=int,
                    help='Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.2, type=float,
                    help='ambiguity level (q)')
parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR-100 fine-grained training')


def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]  ## 0.95,0.8
    iterations = args.lr_decay_epochs.split(',')  ## 700,800,900
    args.lr_decay_epochs = list([])  ## 700,800,900
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # print('distributed:',args.distributed)
    model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_ep_{ep}_ps_{ps}_lw_{lw}_pm_{pm}_arch_{arch}_heir_{heir}_sd_{seed}'.format(
        ds=args.dataset,
        pr=args.partial_rate,
        lr=args.lr,
        ep=args.epochs,
        ps=args.prot_start,
        lw=args.loss_weight,
        pm=args.proto_m,
        arch=args.arch,
        seed=args.seed,
        heir=args.hierarchical)
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    # print('exp_dir',args.exp_dir)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:  ## False+0
        def print_pass(*args):
            pass

        builtins.print = print_pass
    if args.distributed:  ## False
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:  ## False
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = INS(args, SupConResNet)
    print('args.distributed', args.distributed)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch + 1))   #y = 1 /(epoch + 1)
    scheduler_1 = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=2)  #容忍两次还没有更新，开始调整学习率
    scheduler_2 = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=2)  #容忍两次还没有更新，开始调整学习率

    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                            weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.dataset == 'cam16_mil':
        train_loader, train_givenY, train_givenY_pos, train_sampler, test_loader, train_loader_cls, train_givenY_cls,\
        train_sampler_cls, test_bag_loader, patch_label, patch_label_pos, train_bag_loader = load_cam16_mil(batch_size=args.batch_size)        # test_bag_loader, train_sampler_cls,train_bag_loader, train_bag_sampler
    elif args.dataset == 'tcga':
        train_loader, train_givenY, train_givenY_pos, train_sampler, train_loader_cls, train_givenY_cls, \
        train_sampler_cls, test_bag_loader, train_bag_loader = load_TCGA(
            batch_size=args.batch_size)
        # train_bag_loader, test_bag_loader = load_TCGA(1)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    print('Calculating uniform targets...')
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[
        1])  #
    confidence = train_givenY.float() / tempY
    confidence = confidence.cuda()

    tempY_pos = train_givenY_pos.sum(dim=1).unsqueeze(1).repeat(1, train_givenY_pos.shape[
        1])  #
    confidence_pos = train_givenY_pos.float() / tempY_pos
    confidence_pos = confidence_pos.cuda()

    tempY_cls = train_givenY_cls.sum(dim=1).unsqueeze(1).repeat(1, train_givenY_cls.shape[
        1])  #
    confidence_cls = train_givenY_cls.float() / tempY_cls
    confidence_cls = confidence_cls.cuda()

    loss_fn = partial_loss(confidence)
    loss_fn_pos = partial_loss(confidence_pos)
    # loss_fn = Focalloss(confidence)
    loss_cont_fn = SupConLoss()

    # set loss functions (with pseudo-targets maintained)
    if args.gpu == 0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir, 'tensorboard'), flush_secs=2)
    else:
        logger = None

    print('\nStart Training\n')

    epoch = 0
    cls_loss = Cls_loss(confidence_cls.cuda())
    # cls_loss = Focalloss(confidence_cls.cuda())

    # test(model, args, 0, logger,test_bag_loader=test_bag_loader)

    # train_classifier_pretrain(train_loader_cls, model, cls_loss, loss_cont_fn, optimizer, 0, args, logger, False)
    auc_bag, auc_bag2 = test(model, args, epoch, logger,test_bag_loader=test_bag_loader)

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        start_upd_prot = epoch >= args.prot_start  ## epoch>80 True

        adjust_learning_rate(args, optimizer, epoch)

        if epoch == args.prot_start:
            print("################warm up end################")
        if epoch == 0:
            train_classifier_pretrain(train_loader_cls, model, cls_loss, loss_cont_fn, optimizer, scheduler_1, 0, args, logger, False)
        elif epoch>=1 and epoch<args.prot_start:
            train(train_loader, model, loss_fn_pos, loss_cont_fn, optimizer, scheduler_2, epoch, args, logger, start_upd_prot)
            # train_bag(train_bag_loader, model, 0, optimizer, scheduler_1, epoch, args)
        else:
            train(train_loader_cls, model, loss_fn, loss_cont_fn, optimizer, scheduler_2, epoch, args, logger, start_upd_prot)
            train_bag(train_bag_loader, model, 0, optimizer, scheduler_1, epoch, args)

        loss_fn.set_conf_ema_m(epoch, args)

        auc_bag, auc_bag2 = test(model, args, epoch, logger,test_bag_loader=test_bag_loader)

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Bag AUC {}, Bag AUC ins {}.(lr {})\n'.format(epoch
                                                                              , auc_bag,auc_bag2,
                                                                              optimizer.param_groups[0]['lr']))

def train_classifier_pretrain(train_loader, model, loss_fn, loss_cont_fn, optimizer,scheduler, epoch, args, tb_logger,
                              start_upd_prot=False):
    print("the first epoch for classifier train")
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_w, images_s, labels, point, index) in enumerate(
            train_loader):  ## (256,3,32,32)(256,3,32,32)(256,10)(256)(256)
        data_time.update(time.time() - end)
        true_labels = labels[0]
        labels = labels[1]  #
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()

        cls_out = model(images_w, args, eval_only=True)

        loss = loss_fn(cls_out, index)

        loss_cls_log.update(loss.item())
        loss_cont_log.update(0)

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc_proto.update(acc[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)

# 254
def train(train_loader, model, loss_fn, loss_cont_fn, optimizer, scheduler, epoch, args, tb_logger, start_upd_prot=False, target_label=None):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_w, images_s, labels, point, index) in enumerate(
            train_loader):  ## (256,3,32,32)(256,3,32,32)(256,10)(256)(256)

        true_labels = labels[0]
        labels = labels[1]
        data_time.update(time.time() - end)
        for l in range(labels.shape[0]):
            if (labels[l].equal(torch.tensor([0, 1]))):
                labels[l] = torch.tensor([1, 1])
                # print('warm up will not show')

        images_w, images_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()

        cls_out, features_cont, pseudo_score_cont, partial_target_cont, score_prot \
            = model(images_w, images_s, Y, args)
        batch_size = cls_out.shape[0]

        pseudo_target_max, pseudo_target_cont = torch.max(pseudo_score_cont, dim=1)  # 8194,
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)  #

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y, point=point)

            # warm up ended

        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels
        else:
            mask = None
            # Warmup using MoCo

        # contrastive loss
        loss_cont = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        loss_cls = loss_fn(cls_out, index)

        loss = loss_cls + args.loss_weight * loss_cont
        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc = accuracy(score_prot, Y_true)[0]  #
        acc_proto.update(acc[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        # scheduler.step(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)
    # return teacher_auc
def train_bag(train_loader, model, loss_fn, optimizer,scheduler, epoch, args):
    # train_loader is a bag loader: 1xNx3xHxW, weak aug or no aug
    # switch to train mode
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_cls_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    # return bag.float() / 255, [patch_labels, slide_label, idx_slide, slide_name], index
    for i, (images, labels) in enumerate(
            train_loader):  ## (1,N,512)
        # X : N*3*32*32
        # Y : N
        data_time.update(time.time() - end)
        images, Y = images[0].cuda(), labels.cuda()
        bag_pred = model(images, args, bag_flag=True) # 2
        loss = torch.nn.CrossEntropyLoss()
        loss_res = loss(bag_pred,Y)
        # loss = -1.*(Y*torch.log(bag_pred[0,1]+1e-5)+(1.-Y)*torch.log(1.-bag_pred[0,1]+1e-5))
        weight_bag_loss = 1.0
        bag_loss = loss_res * weight_bag_loss
        loss_cls_log.update(bag_loss.item())

        optimizer.zero_grad()
        bag_loss.backward()
        optimizer.step()
        # scheduler.step(bag_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

def test(model, args, epoch, tb_logger, test_bag_loader=None):
    # test adding bag loader
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        AUC = AverageMeter("AUC")
        bag_AUC = AverageMeter("bag_AUC")
        output = AverageMeter("output")
        output_pos = AverageMeter("output_pos")
        pred_score_net = []
        target_net = []
        output_net = []
        pred_score_net_bag = []
        target_net_bag = []
        pred_score_net_ins = []
        target_net_ins = []
        pred_score_net_ins2 = []
        target_net_ins2 = []
        pred_score_net_ins3 = []
        target_net_ins3 = []
        # top5_acc = AverageMeter("Top5")
        # test_bag 加入attention pooling
        for batch_idx, (images, labels) in enumerate(test_bag_loader):
            # 1 bag: N*3*H*w
            # images = torch.squeeze(images,0)
            labels = labels.cuda()
            # print('labels:',labels)
            outputs = torch.zeros(images.shape[1],2)
            # print('images',images.shape)
            images = images.cuda()
            images = torch.squeeze(images,0)# 1*N*dim ->N*Dim
            # images = images.view(images.shape[0], 32, -1)
            # images = torch.repeat_interleave(images.unsqueeze(dim=1), repeats=3, dim=1)
            output_res = model(images, args, bag_flag=True)
            output_ins = model(images, args, eval_only=True)

            output_ins1 = torch.mean(output_ins[:, 1], dim=0, keepdim=True)
            output_ins2,_ = torch.max(output_ins[:, 1], dim=0, keepdim=True)
            output_ins3 = torch.logsumexp(output_ins[:, 1], dim=0, keepdim=True)

            # print(output_res)
            pred_score_net_bag += output_res[:,1].tolist()
            target_net_bag += labels.tolist()
            
            pred_score_net_ins += output_ins1.tolist()
            target_net_ins += labels.tolist()
            pred_score_net_ins2 += output_ins2.tolist()
            target_net_ins2 += labels.tolist()
            pred_score_net_ins3 += output_ins3.tolist()
            target_net_ins3 += labels.tolist()
            # print("bag output:", output_res)
        
        pred_score = outputs[:, 1]
        auc_bag = metrics.roc_auc_score(target_net_bag, pred_score_net_bag)
        bag_AUC.update(auc_bag)
        
        auc_bag_ins = metrics.roc_auc_score(target_net_ins, pred_score_net_ins)
        auc_bag_ins2 = metrics.roc_auc_score(target_net_ins2, pred_score_net_ins2)
        auc_bag_ins3 = metrics.roc_auc_score(target_net_ins3, pred_score_net_ins2)

        # print(output)

        print(' Auc_bag is ', auc_bag,' Auc_bag_ins ',auc_bag_ins,' Auc_bag_ins2 ',auc_bag_ins2,' Auc_bag_ins3 ',auc_bag_ins3)

    return  auc_bag, auc_bag_ins


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
