import random

import torch
from torch.utils.data import Dataset
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image

from six.moves import cPickle as pickle
import os
import platform
from tqdm import tqdm
import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

# from .randaugment import RandomAugment
# from .randaugment import RandomAugment

bag_length = 100
pos_slide_ratio = 0.5

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)



def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3072).reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs, axis=0)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data():
    # Load the raw CIFAR-10 data
    cifar10_dir = '/home/guest/Desktop/PiCO/PICO-origin/data/cifar-10-batches-py'
    x_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    x_train, y_train, X_test, y_test = torch.from_numpy(x_train), torch.from_numpy(y_train), \
                                       torch.from_numpy(X_test), torch.from_numpy(y_test)
    return x_train, y_train, X_test, y_test

def random_shuffle(input_tensor):
    random.seed(0)
    length = input_tensor.shape[0]
    random_idx = torch.randperm(length)
    output_tensor = input_tensor[random_idx]
    print("#################shuffle#################")
    print(output_tensor)
    return output_tensor

class CIFAR_WholeSlide_challenge(torch.utils.data.Dataset):
    def __init__(self, train, positive_num=[9], negative_num=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 bag_length=10, return_bag=False, num_img_per_slide=600, pos_patch_ratio=0.1, pos_slide_ratio=0.5, transform=None, accompanyPos=True, pretrain=False,
                 idx_all_slides=None, label_all_slides=None):
        self.train = train
        self.positive_num = positive_num  # transform the N-class into 2-class
        self.negative_num = negative_num  # transform the N-class into 2-class
        self.bag_length = bag_length
        self.return_bag = return_bag  # return patch ot bag
        self.transform = transform    # transform the patch image
        self.num_img_per_slide = num_img_per_slide
        self.pretrain = pretrain

        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8), ## more than strong_transform
            transforms.RandomGrayscale(p=0.2), ## more than strong_transform
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            # RandomAugment(3, 5), ## more than weak_transform
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        if train:
            self.ds_data, self.ds_label, _, _ = get_CIFAR10_data()
            try:
                self.ds_data_simCLR_feat = torch.from_numpy(np.load("./Datasets_loader/all_feats_CIFAR.npy")[:50000, :]).float()
                print("Pre-trained feat found")
            except:
                print("No pre-trained feat found")

        self.build_whole_slides(num_img=num_img_per_slide, positive_nums=positive_num, negative_nums=negative_num, pos_patch_ratio=pos_patch_ratio, pos_slide_ratio=pos_slide_ratio)

        print("")

    def build_whole_slides(self, num_img, positive_nums, negative_nums, pos_patch_ratio=0.1, pos_slide_ratio=0.5):
        # num_img: num of images per slide
        # positive patch ratio in each slide
        num_pos_per_slide = int(num_img * pos_patch_ratio)
        num_neg_per_slide = num_img - num_pos_per_slide
        print(num_pos_per_slide, num_neg_per_slide)

        idx_pos = []
        for num in positive_nums:
            idx_pos.append(torch.where(self.ds_label == num)[0])
        idx_pos = torch.cat(idx_pos).unsqueeze(1)
        idx_neg = []
        for num in negative_nums:
            idx_neg.append(torch.where(self.ds_label == num)[0])
        idx_neg = torch.cat(idx_neg).unsqueeze(1)

        idx_pos = random_shuffle(idx_pos)
        idx_neg = random_shuffle(idx_neg)

        # build pos slides using calculated
        num_pos_2PosSlides = int(idx_neg.numel() // ((1 - pos_slide_ratio) / (pos_patch_ratio*pos_slide_ratio) + (1 - pos_patch_ratio) / pos_patch_ratio))
        if num_pos_2PosSlides > idx_pos.shape[0]:
            num_pos_2PosSlides = idx_pos.shape[0]
        num_pos_2PosSlides = int(num_pos_2PosSlides // num_pos_per_slide * num_pos_per_slide)
        num_neg_2PosSlides = int(num_pos_2PosSlides * ((1-pos_patch_ratio)/pos_patch_ratio))
        num_neg_2NegSlides = int(num_pos_2PosSlides * ((1-pos_slide_ratio)/(pos_patch_ratio*pos_slide_ratio)))

        num_neg_2PosSlides = int(num_neg_2PosSlides // num_neg_per_slide * num_neg_per_slide)
        num_neg_2NegSlides = int(num_neg_2NegSlides // num_img * num_img)

        if num_neg_2PosSlides // num_neg_per_slide != num_pos_2PosSlides // num_pos_per_slide :
            num_diff_slide = num_pos_2PosSlides // num_pos_per_slide - num_neg_2PosSlides // num_neg_per_slide
            num_pos_2PosSlides = num_pos_2PosSlides - num_pos_per_slide * num_diff_slide

        idx_pos = idx_pos[0:num_pos_2PosSlides]
        idx_neg = idx_neg[0:(num_neg_2PosSlides+num_neg_2NegSlides)]

        idx_pos_toPosSlide = idx_pos[:].reshape(-1, num_pos_per_slide)
        idx_neg_toPosSlide = idx_neg[0:num_neg_2PosSlides].reshape(-1, num_neg_per_slide)
        idx_neg_toNegSlide = idx_neg[num_neg_2PosSlides:].reshape(-1, num_img)

        idx_pos_slides = torch.cat([idx_pos_toPosSlide, idx_neg_toPosSlide], dim=1)
        # idx_pos_slides = idx_pos_slides[:, torch.randperm(idx_pos_slides.shape[1])]  #  shuffle pos and neg idx
        for i_ in range(idx_pos_slides.shape[0]):
            idx_pos_slides[i_, :] = idx_pos_slides[i_, torch.randperm(idx_pos_slides.shape[1])]
        # idx_neg_slides = idx_neg_toNegSlide
        self.idx_neg_slides = idx_neg_toNegSlide
        self.idx_pos_slides = idx_pos_slides
        self.idx_all_slides = torch.cat([self.idx_pos_slides, self.idx_neg_slides], dim=0)
        self.label_all_slides = torch.cat([torch.ones(self.idx_pos_slides.shape[0]), torch.zeros(self.idx_neg_slides.shape[0])],
                                          dim=0)
        self.label_all_slides = self.label_all_slides.unsqueeze(1).repeat([1, self.idx_all_slides.shape[1]]).long()
        print("[Info] dataset: {}".format(self.idx_all_slides.shape))
        print(self.idx_all_slides.numel())
        self.idx_all_slides_pos = torch.cat([self.idx_pos_slides], dim=0)
        self.label_all_slides_pos = torch.cat([torch.ones(self.idx_pos_slides.shape[0])], dim=0)
        self.label_all_slides_pos = self.label_all_slides_pos.unsqueeze(1).repeat([1, self.idx_all_slides_pos.shape[1]]).long()

        #self.visualize(idx_pos_slides[0])

    def __getitem__(self, index):
        # 如果epoch = 0 return neg[1,0] pos [0,1],point
        # 如果1 <epoch < warmup pos[1,1],point
        # 如果epoch>=warmup return neg,pos
        idx_image = self.idx_all_slides_pos.flatten()[index]
        slide_label = self.label_all_slides_pos.flatten()[index]
        idx_slide = index // self.num_img_per_slide
        slide_name = str(idx_slide)
        patch = self.ds_data[idx_image] # 3,32,32
        patch_label = self.ds_label[idx_image]
        patch_label = int(patch_label in self.positive_num) # equal to each_true_label
        # patch = patch.float()/255
        each_image_w = self.weak_transform(patch)
        each_image_s = self.strong_transform(patch)
        # neg
        point = -1
        if slide_label == 0:
            point = 0
            each_label = torch.tensor([1, 0])
        # pos
        else:
            point = 1
            each_label = torch.tensor([1, 1]) # 暂时确定
            # 在train的时候进行判断如果epoch==0 就将point=1的label改成[0,1]
        # label:1 [1,1]
        return each_image_w, each_image_s, each_label, patch_label, point, index
        # return patch.float()/255, [patch_label, slide_label, idx_slide, slide_name], index,p

    def __len__(self):
        if self.return_bag:
            return self.idx_all_slides_pos.shape[1] // self.bag_length * self.idx_all_slides_pos.shape[0]
        else:
            if not self.train:
                return self.idx_all_slides.numel()
            else:
                # 如果是train
                return self.idx_all_slides_pos.numel()

class CIFAR_WholeSlide_challenge_cls(torch.utils.data.Dataset):
    def __init__(self, train, positive_num=[8, 9], negative_num=[0, 1, 2, 3, 4, 5, 6, 7],
                 bag_length=10, return_bag=False, num_img_per_slide=100, pos_patch_ratio=0.1, pos_slide_ratio=0.5, transform=None, accompanyPos=True, pretrain=False,
                 idx_all_slides=None, label_all_slides=None):
        self.train = train
        self.positive_num = positive_num  # transform the N-class into 2-class
        self.negative_num = negative_num  # transform the N-class into 2-class
        self.bag_length = bag_length
        self.return_bag = return_bag  # return patch ot bag
        self.transform = transform    # transform the patch image
        self.num_img_per_slide = num_img_per_slide
        self.pretrain = pretrain
        self.idx_all_slides = idx_all_slides
        self.label_all_slides = label_all_slides
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8), ## more than strong_transform
            transforms.RandomGrayscale(p=0.2), ## more than strong_transform
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            # RandomAugment(3, 5), ## more than weak_transform
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        if train:
            self.ds_data, self.ds_label, _, _ = get_CIFAR10_data()
            try:
                self.ds_data_simCLR_feat = torch.from_numpy(np.load("./Datasets_loader/all_feats_CIFAR.npy")[:50000, :]).float()
                print("Pre-trained feat found")
            except:
                print("No pre-trained feat found")

    def __getitem__(self, index):
        # 如果epoch = 0 return neg[1,0] pos [0,1],point
        # 如果1 <epoch < warmup pos[1,1],point
        # 如果epoch>=warmup return neg,pos
        # instance-level
        if not self.return_bag:
            idx_image = self.idx_all_slides.flatten()[index]
            slide_label = self.label_all_slides.flatten()[index]
            idx_slide = index // self.num_img_per_slide
            slide_name = str(idx_slide)
            patch = self.ds_data[idx_image] # 3,32,32
            patch_label = self.ds_label[idx_image]
            patch_label = int(patch_label in self.positive_num) # equal to each_true_label
            # patch = patch.float()/255
            each_image_w = self.weak_transform(patch)
            each_image_s = self.strong_transform(patch)
            # neg
            point = -1
            if slide_label == 0:
                point = 0
                each_label = torch.tensor([1, 0])
            # pos
            else:
                point = 1
                each_label = torch.tensor([0, 1]) # 暂时确定
                # 在train的时候进行判断如果epoch==0 就将point=1的label改成[0,1]
            # label:1 [1,1]
            return each_image_w, each_image_s, each_label, patch_label, point, index
            # return patch.float()/255, [patch_label, slide_label, idx_slide, slide_name], index,p
        # bag-level
        else:
            bagPerSlide = self.idx_all_slides.shape[1] // self.bag_length
            idx_slide = index // bagPerSlide
            idx_bag_in_slide = index % bagPerSlide
            idx_images = self.idx_all_slides[idx_slide,
                         (idx_bag_in_slide * self.bag_length):((idx_bag_in_slide + 1) * self.bag_length)]
            bag = self.ds_data[idx_images]
            for i in range(bag.shape[0]):
                bag[i] = self.weak_transform(bag[i])
            patch_labels_raw = self.ds_label[idx_images]
            patch_labels = torch.zeros_like(patch_labels_raw)
            for num in self.positive_num:
                patch_labels[patch_labels_raw == num] = 1
            patch_labels = patch_labels.long()
            slide_label = self.label_all_slides[idx_slide, 0]
            slide_name = str(idx_slide)
            return bag.float() / 255, [patch_labels, slide_label, idx_slide, slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.idx_all_slides.shape[1] // self.bag_length * self.idx_all_slides.shape[0]
        else:
            return self.idx_all_slides.numel()

class CIFAR_WholeSlide_challenge_val(torch.utils.data.Dataset):
    def __init__(self, train, positive_num=[8, 9], negative_num=[0, 1, 2, 3, 4, 5, 6, 7],
                 bag_length=100, return_bag=False, num_img_per_slide=100, pos_patch_ratio=0.1, pos_slide_ratio=0.5, transform=None, accompanyPos=True, pretrain=False,
                 idx_all_slides=None, label_all_slides=None):
        self.train = train
        self.positive_num = positive_num  # transform the N-class into 2-class
        self.negative_num = negative_num  # transform the N-class into 2-class
        self.return_bag = return_bag
        self.bag_length = bag_length
        # self.transform = transform  # transform the patch image
        self.num_img_per_slide = num_img_per_slide
        self.pretrain = pretrain
        self.idx_all_slides = idx_all_slides
        self.label_all_slides = label_all_slides
        self.test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        _, _ , self.ds_data, self.ds_label = get_CIFAR10_data()
        self.build_whole_slides(num_img=num_img_per_slide, positive_nums=positive_num, negative_nums=negative_num, pos_patch_ratio=pos_patch_ratio, pos_slide_ratio=pos_slide_ratio)

    def build_whole_slides(self, num_img, positive_nums, negative_nums, pos_patch_ratio=0.1, pos_slide_ratio=0.5):
        # num_img: num of images per slide
        # positive patch ratio in each slide
        num_pos_per_slide = int(num_img * pos_patch_ratio)
        num_neg_per_slide = num_img - num_pos_per_slide
        print(num_pos_per_slide, num_neg_per_slide)

        idx_pos = []
        for num in positive_nums:
            idx_pos.append(torch.where(self.ds_label == num)[0])
        idx_pos = torch.cat(idx_pos).unsqueeze(1)
        idx_neg = []
        for num in negative_nums:
            idx_neg.append(torch.where(self.ds_label == num)[0])
        idx_neg = torch.cat(idx_neg).unsqueeze(1)

        idx_pos = random_shuffle(idx_pos)
        idx_neg = random_shuffle(idx_neg)

        # build pos slides using calculated
        num_pos_2PosSlides = int(idx_neg.numel() // ((1 - pos_slide_ratio) / (pos_patch_ratio*pos_slide_ratio) + (1 - pos_patch_ratio) / pos_patch_ratio))
        if num_pos_2PosSlides > idx_pos.shape[0]:
            num_pos_2PosSlides = idx_pos.shape[0]
        num_pos_2PosSlides = int(num_pos_2PosSlides // num_pos_per_slide * num_pos_per_slide)
        num_neg_2PosSlides = int(num_pos_2PosSlides * ((1-pos_patch_ratio)/pos_patch_ratio))
        num_neg_2NegSlides = int(num_pos_2PosSlides * ((1-pos_slide_ratio)/(pos_patch_ratio*pos_slide_ratio)))

        num_neg_2PosSlides = int(num_neg_2PosSlides // num_neg_per_slide * num_neg_per_slide)
        num_neg_2NegSlides = int(num_neg_2NegSlides // num_img * num_img)

        if num_neg_2PosSlides // num_neg_per_slide != num_pos_2PosSlides // num_pos_per_slide :
            num_diff_slide = num_pos_2PosSlides // num_pos_per_slide - num_neg_2PosSlides // num_neg_per_slide
            num_pos_2PosSlides = num_pos_2PosSlides - num_pos_per_slide * num_diff_slide

        idx_pos = idx_pos[0:num_pos_2PosSlides]
        idx_neg = idx_neg[0:(num_neg_2PosSlides+num_neg_2NegSlides)]

        idx_pos_toPosSlide = idx_pos[:].reshape(-1, num_pos_per_slide)
        idx_neg_toPosSlide = idx_neg[0:num_neg_2PosSlides].reshape(-1, num_neg_per_slide)
        idx_neg_toNegSlide = idx_neg[num_neg_2PosSlides:].reshape(-1, num_img)

        idx_pos_slides = torch.cat([idx_pos_toPosSlide, idx_neg_toPosSlide], dim=1)
        # idx_pos_slides = idx_pos_slides[:, torch.randperm(idx_pos_slides.shape[1])]  #  shuffle pos and neg idx
        for i_ in range(idx_pos_slides.shape[0]):
            idx_pos_slides[i_, :] = idx_pos_slides[i_, torch.randperm(idx_pos_slides.shape[1])]
        # idx_neg_slides = idx_neg_toNegSlide
        self.idx_neg_slides = idx_neg_toNegSlide
        self.idx_pos_slides = idx_pos_slides
        self.idx_all_slides = torch.cat([self.idx_pos_slides, self.idx_neg_slides], dim=0)
        self.label_all_slides = torch.cat([torch.ones(self.idx_pos_slides.shape[0]), torch.zeros(self.idx_neg_slides.shape[0])],
                                          dim=0)
        self.label_all_slides = self.label_all_slides.unsqueeze(1).repeat([1, self.idx_all_slides.shape[1]]).long()
        print("[Info] dataset: {}".format(self.idx_all_slides.shape))
        print(self.idx_all_slides.numel())
        self.idx_all_slides_pos = torch.cat([self.idx_pos_slides], dim=0)
        self.label_all_slides_pos = torch.cat([torch.ones(self.idx_pos_slides.shape[0])], dim=0)
        self.label_all_slides_pos = self.label_all_slides_pos.unsqueeze(1).repeat([1, self.idx_all_slides_pos.shape[1]]).long()

    def __getitem__(self, index):
        # instance-level
        if not self.return_bag:
            idx_image = self.ds_data[index]
            patch_label = self.ds_label[index]
            patch_label = int(patch_label in self.positive_num)  # equal to each_true_label
            patch = self.test_transform(idx_image)
            return patch, patch_label, index
        # bag-level
        else:
            bagPerSlide = self.idx_all_slides.shape[1] // self.bag_length
            idx_slide = index // bagPerSlide
            idx_bag_in_slide = index % bagPerSlide
            idx_images = self.idx_all_slides[idx_slide,
                         (idx_bag_in_slide * self.bag_length):((idx_bag_in_slide + 1) * self.bag_length)]
            bag = self.ds_data[idx_images]
            for i in range(bag.shape[0]):
                bag[i] = self.test_transform(bag[i])
            patch_labels_raw = self.ds_label[idx_images]
            patch_labels = torch.zeros_like(patch_labels_raw)
            for num in self.positive_num:
                patch_labels[patch_labels_raw == num] = 1
            patch_labels = patch_labels.long()
            slide_label = self.label_all_slides[idx_slide, 0]
            slide_name = str(idx_slide)
            return bag.float() / 255, [patch_labels, slide_label, idx_slide, slide_name], index

    def __len__(self):
        if not self.return_bag:
            return len(self.ds_data)
        else:
            return self.idx_all_slides.shape[1] // self.bag_length * self.idx_all_slides.shape[0]

def load_cifarmil(partial_rate, batch_size, pretrain):
    # args = get_parser()

    pos_patch_ratio = partial_rate

    print("=========== pos patch ratio: {} ===========".format(pos_patch_ratio))
    positive_num = [9]
    negative_num = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ds_data, ds_label, _, _ = get_CIFAR10_data()

    train_ds_return_instance = CIFAR_WholeSlide_challenge(train=True, positive_num=positive_num,
                                                          negative_num=negative_num,
                                                          bag_length=bag_length, return_bag=False,
                                                          num_img_per_slide=bag_length,
                                                          pos_patch_ratio=pos_patch_ratio,
                                                          pos_slide_ratio=pos_slide_ratio,
                                                          transform=None, pretrain= pretrain)
    # Partial_Y = train_ds_return_instance.build_whole_slides(num_img = args.bag_length, positive_nums=positive_num, negative_nums=positive_num, pos_patch_ratio=0.1, pos_slide_ratio=0.5)
    # s = train_ds_return_instance.label_all_slides
    # train的2个伪标签集构造
    
    slide_label = train_ds_return_instance.label_all_slides.flatten() # 构造Partial_Y 43900个样本
    slide_label_pos = train_ds_return_instance.label_all_slides_pos.flatten()
    partialY = torch.zeros(slide_label.shape[0], 2) # 二分类问题
    partialY_cls = torch.zeros(slide_label.shape[0], 2)  # 二分类问题 第一次train
    # partialY_after_warm = torch.zeros(slide_label.shape[0], 2)  # 二分类问题 第一次train
    # partialY_warmup = torch.zeros(slide_label_pos.shape[0],2)
    # for i in range(slide_label_pos.shape[0]):
    #     partialY_warmup[i] = torch.tensor([1, 1])
    for i in range(slide_label.shape[0]):
        if slide_label[i] == 0: # 只有这一种 训练的很差 加入病理数据会不会好一些
            # print("negative")
            partialY[i] = torch.tensor([1, 0])
            partialY_cls[i] = torch.tensor([1, 0])
        else:
            partialY[i] = torch.tensor([1, 1])
            partialY_cls[i] = torch.tensor([0, 1])

    idx_all_slides_cls = train_ds_return_instance.idx_all_slides
    label_all_slides_cls = train_ds_return_instance.label_all_slides
    # 都是对应的
    train_ds_return_instance_cls = CIFAR_WholeSlide_challenge_cls(train=True, positive_num=positive_num,
                                                          negative_num=negative_num,
                                                          bag_length=bag_length, return_bag=False,
                                                          num_img_per_slide=bag_length,
                                                          pos_patch_ratio=pos_patch_ratio,
                                                          pos_slide_ratio=pos_slide_ratio,
                                                          transform=None, pretrain=pretrain,
                                                          idx_all_slides=idx_all_slides_cls, label_all_slides=label_all_slides_cls)
    train_ds_return_bag_cls = CIFAR_WholeSlide_challenge_cls(train=True, positive_num=positive_num,
                                                          negative_num=negative_num,
                                                          bag_length=bag_length, return_bag=True,
                                                          num_img_per_slide=bag_length,
                                                          pos_patch_ratio=pos_patch_ratio,
                                                          pos_slide_ratio=pos_slide_ratio,
                                                          transform=None, pretrain=pretrain,
                                                          idx_all_slides=idx_all_slides_cls, label_all_slides=label_all_slides_cls)
    val_ds_return_instance = CIFAR_WholeSlide_challenge_val(train=False, positive_num=positive_num,
                                                        negative_num=negative_num, bag_length=bag_length,return_bag=False, pos_patch_ratio=pos_patch_ratio)
    val_ds_return_bag = CIFAR_WholeSlide_challenge_val(train=False, positive_num=positive_num,
                                                            negative_num=negative_num,
                                                            bag_length=bag_length, return_bag=True,
                                                            num_img_per_slide=bag_length,
                                                            pos_patch_ratio=pos_patch_ratio, pos_slide_ratio=pos_slide_ratio)
    # val_ds_return_instance = CIFAR_WholeSlide_challenge(train=False, positive_num=positive_num,
    #                                                     negative_num=negative_num,
    #                                                     bag_length=bag_length, return_bag=False,
    #                                                     num_img_per_slide=bag_length,
    #                                                     pos_patch_ratio=pos_patch_ratio,
    #                                                     pos_slide_ratio=pos_slide_ratio,
    #                                                     transform=None, pretrain=pretrain
    #                                                     ,idx_all_slides=idx_all_slides_cls, label_all_slides=label_all_slides_cls)

    train_loader_instance = torch.utils.data.DataLoader(train_ds_return_instance, batch_size=batch_size,
                                                        shuffle=True, num_workers=4, drop_last=False)

    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=batch_size,
                                                      shuffle=False, num_workers=4, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag,batch_size=1,shuffle=False,num_workers=4,drop_last=False)
    # 有时间加入sampler
    # val_sampler = torch.utils.data.distributed.DistributedSampler(train_ds_return_instance)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
    #     sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    print('Average candidate num: ', partialY.sum(1).mean())  ## 1.9 表示平均每个样本有的标签个数均值

    # for i, (images_w, images_s, labels, true_labels, point, index) in enumerate(tqdm(val_loader_instance, desc='Instance Training')):
    #     # patch.float()/255, [patch_label, slide_label, idx_slide, slide_name], index
    #     print(labels, true_labels, point)
    #     break
    # for batch_idx, (images, labels, index) in enumerate(val_loader_instance):
    #     print(labels)
        # break
    # print("#################################")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds_return_instance)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_ds_return_instance,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler is None),
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              sampler=train_sampler,
                                                              drop_last=True)
    
    train_sampler_bag = torch.utils.data.distributed.DistributedSampler(train_ds_return_bag_cls)
    partial_matrix_train_loader_bag = torch.utils.data.DataLoader(dataset=train_ds_return_bag_cls,
                                                              batch_size=1,
                                                              shuffle=(train_sampler_bag is None),
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              sampler=train_sampler_bag,
                                                              drop_last=True)
    
    # test_loader = torch.utils.data.DataLoader(dataset=val_ds_return_instance, batch_size=batch_size, shuffle=False, num_workers=4,
    #     sampler=torch.utils.data.distributed.DistributedSampler(val_ds_return_instance, shuffle=False))
    # return train_loader, train_givenY, train_sampler, test_loader
    train_sampler_cls = torch.utils.data.distributed.DistributedSampler(train_ds_return_instance_cls)
    partial_matrix_train_loader_cls = torch.utils.data.DataLoader(dataset=train_ds_return_instance_cls,
                                                                  batch_size=batch_size,
                                                                  shuffle=(train_sampler_cls is None),
                                                                  num_workers=4,
                                                                  pin_memory=True,
                                                                  sampler=train_sampler_cls,
                                                                  drop_last=True)
    return partial_matrix_train_loader, partialY, train_sampler, val_loader_instance, \
partial_matrix_train_loader_cls, partialY_cls, val_loader_bag, train_sampler_cls,\
    partial_matrix_train_loader_bag, train_sampler_bag


# res = load_cifarmil(0.05,2)
# print(res)
# res = load_cifarmil(0.05,2)
# print(res)
if __name__ == '__main__':
    train_loader, train_givenY, train_sampler, test_loader, train_loader_cls, train_givenY_cls, test_bag_loader, train_sampler_cls, train_bag_loader, train_bag_sampler = load_cifarmil(
        partial_rate=0.5,
        batch_size=256,
        pretrain=False)
    # find the bag with lower score
