import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
import glob
from skimage import io
from tqdm import tqdm
from utils.randaugment import RandomAugment

# 1. give true labels
# 2. warm-up only give positive bag
# 3. train positive lables
# only give pos instances

def statistics_slide(slide_path_list):
    num_pos_patch_allPosSlide = 0
    num_patch_allPosSlide = 0
    num_neg_patch_allNegSlide = 0
    num_all_slide = len(slide_path_list)
    for i in slide_path_list:
        if 'pos' in i.split('/')[-1]:  # pos slide
            num_pos_patch = len(glob.glob(i + "/*_pos.jpg"))
            num_patch = len(glob.glob(i + "/*.jpg"))
            num_pos_patch_allPosSlide = num_pos_patch_allPosSlide + num_pos_patch
            num_patch_allPosSlide = num_patch_allPosSlide + num_patch
            # print(i,num_pos_patch,num_patch)
        else:  # neg slide
            num_neg_patch = len(glob.glob(i + "/*.jpg"))
            num_neg_patch_allNegSlide = num_neg_patch_allNegSlide + num_neg_patch
    print("num_pos_patch_allPosSlide:",num_pos_patch_allPosSlide)
    print("[DATA INFO] {} slides totally".format(num_all_slide))
    print("[DATA INFO] pos_patch_ratio in pos slide: {:.4f}({}/{})".format(
        num_pos_patch_allPosSlide / num_patch_allPosSlide, num_pos_patch_allPosSlide, num_patch_allPosSlide))
    print("[DATA INFO] num of patches: {} ({} from pos slide, {} from neg slide)".format(
        num_patch_allPosSlide+num_neg_patch_allNegSlide, num_patch_allPosSlide, num_neg_patch_allNegSlide))

    return num_patch_allPosSlide+num_neg_patch_allNegSlide


class CAMELYON_16_feat(torch.utils.data.Dataset):
    # @profile
    def __init__(self, root_dir='./patches_byDSMIL',
                 train=True, transform=None, downsample=1.0, drop_threshold=0.0, preload=True, return_bag=False, only_pos=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.downsample = downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        self.return_bag = return_bag
        self.only_pos = only_pos
        # self.weak_transform = transforms.Compose(
        #     [
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomApply([
        #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #         ], p=0.8),  ## more than strong_transform
        #         transforms.RandomGrayscale(p=0.2),  ## more than strong_transform
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        # self.strong_transform = transforms.Compose(
        #     [
        #         transforms.RandomHorizontalFlip(),
        #         RandomAugment(3, 5),  ## more than weak_transform
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        if train:
            self.root_dir = os.path.join(self.root_dir, "training")
        else:
            self.root_dir = os.path.join(self.root_dir, "testing")

        all_slides = glob.glob(self.root_dir + "/*")
        # 1.filter the pos slides which have 0 pos patch
        all_pos_slides = glob.glob(self.root_dir + "/*_pos")
        all_pos_slides_tmp = all_pos_slides.copy()
        for i in all_pos_slides:
            num_pos_patch = len(glob.glob(i + "/*_pos.jpg"))
            num_patch = len(glob.glob(i + "/*.jpg"))
            if num_pos_patch/num_patch <= self.drop_threshold:
                all_slides.remove(i)
                all_pos_slides_tmp.remove(i)
                print("[DATA] {} of positive patch ratio {:.4f}({}/{}) is removed".format(
                    i, num_pos_patch/num_patch, num_pos_patch, num_patch))
        statistics_slide(all_slides)
        # 1.1 down sample the slides
        print("================ Down sample ================")
        print("len_pos_slides:", len(all_pos_slides))
        if only_pos:
            all_slides = all_pos_slides_tmp.copy()

        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        self.num_patches = statistics_slide(all_slides)

        save_path = "./Camelyon16_simclrfeats"
        # 2. load all pre-trained patch features (by SimCLR in DSMIL)
        all_slides_name = [i.split('/')[-1] for i in all_slides]

        if train:
            all_slides_feat_file = glob.glob(save_path+"/training/*")
        else:
            all_slides_feat_file = glob.glob(save_path+"/testing/*")

        self.slide_feat_all = np.zeros([self.num_patches, 512], dtype=np.float32)
        self.slide_patch_label_all = np.zeros([self.num_patches], dtype=np.compat.long)
        self.patch_corresponding_slide_label = np.zeros([self.num_patches], dtype=np.compat.long)
        self.patch_corresponding_slide_index = np.zeros([self.num_patches], dtype=np.compat.long)
        self.patch_corresponding_slide_name = np.zeros([self.num_patches], dtype='<U13')
        cnt_slide = 0
        pointer = 0

        for i in all_slides_feat_file:
            slide_name_i = i.split('/')[-1].split('.')[0]
            if slide_name_i not in all_slides_name:
                continue
            slide_i_label_feat = np.load(i)
            slide_i_patch_label = slide_i_label_feat[:, 0]
            slide_i_feat = slide_i_label_feat[:, 1:]
            num_patches_i = slide_i_label_feat.shape[0]

            self.slide_feat_all[pointer:pointer+num_patches_i, :] = slide_i_feat
            self.slide_patch_label_all[pointer:pointer+num_patches_i] = slide_i_patch_label
            self.patch_corresponding_slide_label[pointer:pointer+num_patches_i] = int('pos' in slide_name_i) * np.ones([num_patches_i], dtype=np.compat.long)
            self.patch_corresponding_slide_index[pointer:pointer+num_patches_i] = cnt_slide * np.ones([num_patches_i], dtype=np.compat.long)
            self.patch_corresponding_slide_name[pointer:pointer+num_patches_i] = np.array(slide_name_i).repeat(num_patches_i)
            pointer = pointer + num_patches_i
            cnt_slide = cnt_slide + 1

        self.all_patches = self.slide_feat_all
        self.patch_label = self.slide_patch_label_all
        # print(self.patch_label[14753])
        print("")

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
            self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            bag = self.all_patches[idx_patch_from_slide_i, :]
            patch_labels = self.slide_patch_label_all[idx_patch_from_slide_i]
            slide_label = patch_labels.max()
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            patch_image = self.all_patches[index]
            # each_image_w = self.weak_transform(patch_image)
            # each_image_s = self.strong_transform(patch_image)
            patch_image = torch.from_numpy(patch_image)
            each_image_w = torch.nn.functional.dropout(patch_image,p=0.2,training=False)
            each_image_s = torch.nn.functional.dropout(patch_image, p=0.4, training=False)
            patch_label = self.patch_label[index]
            point = 0
            if self.only_pos:
                # only pos bag
                each_label = torch.tensor([1, 1])
                point = 1
            else:
                # give label
                if patch_label == 0:
                    each_label = torch.tensor([1, 0])
                    point = 0
                else:
                    each_label = torch.tensor([0, 1])
                    point = 1

            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            if not self.train:
                return patch_image, patch_image, [patch_label, each_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], point, index
            return each_image_w,each_image_s, [patch_label, each_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], point, index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches

def load_cam16_mil(batch_size=64):
    train_ds_return_pos_instance = CAMELYON_16_feat(train=True, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=False, only_pos=True)
    train_ds_return_true_instance = CAMELYON_16_feat(train=True, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=False, only_pos=False)
    val_ds_return_instance = CAMELYON_16_feat(train=False, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=False, only_pos=False)
    val_ds_return_bag = CAMELYON_16_feat(train=False, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=True, only_pos=False)
    train_ds_return_bag = CAMELYON_16_feat(train=True, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=True, only_pos=False)

    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=64, shuffle=False,
                                                      num_workers=0, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False,
                                                      num_workers=0, drop_last=False)
    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=False,
                                                      num_workers=0, drop_last=False)
    # for batch_idx, (images_w, images_s, labels, index) in enumerate(val_loader_instance):
    #     print(batch_idx)
    train_sampler_pos = torch.utils.data.distributed.DistributedSampler(train_ds_return_pos_instance)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_ds_return_pos_instance,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler_pos is None),
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              sampler=train_sampler_pos,
                                                              drop_last=True)
    # return train_loader, train_givenY, train_sampler, test_loader
    train_sampler_cls = torch.utils.data.distributed.DistributedSampler(train_ds_return_true_instance)
    partial_matrix_train_loader_cls = torch.utils.data.DataLoader(dataset=train_ds_return_true_instance,
                                                                  batch_size=batch_size,
                                                                  shuffle=(train_sampler_cls is None),
                                                                  num_workers=4,
                                                                  pin_memory=True,
                                                                  sampler=train_sampler_cls,
                                                                  drop_last=True)
    # prepare partial label
    slide_pos_label = train_ds_return_pos_instance.patch_corresponding_slide_label
    partialY_pos = torch.zeros(slide_pos_label.shape[0], 2)
    for i in range(slide_pos_label.shape[0]):
        partialY_pos[i] = torch.tensor([1, 1])

    slide_true_label = train_ds_return_true_instance.patch_corresponding_slide_label
    partialY_cls = torch.zeros(slide_true_label.shape[0], 2)
    partialY = torch.zeros(slide_true_label.shape[0], 2)

    for i in range(slide_true_label.shape[0]):
        if slide_true_label[i] == 0:
            # print("negative")
            partialY[i] = torch.tensor([1, 0])
            partialY_cls[i] = torch.tensor([1, 0])
        else:
            partialY[i] = torch.tensor([1, 1])
            partialY_cls[i] = torch.tensor([0, 1])
    print('Average candidate num: ', partialY.sum(1).mean())

    return partial_matrix_train_loader, partialY, partialY_pos, train_sampler_pos, \
           val_loader_instance,\
           partial_matrix_train_loader_cls, partialY_cls, train_sampler_cls,\
            val_loader_bag,\
            train_ds_return_true_instance.patch_label, train_ds_return_pos_instance.patch_label, train_loader_bag

if __name__ == '__main__':
    # print('test')
    # load_cam16_mil(32)
    transform_data = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x
            # transforms.Normalize(mean=[0.64715815, 0.48541722, 0.65863925], std=[0.24745935, 0.2785922, 0.22133236])])  # CAMELYON16_224x224_5x
    train_ds = CAMELYON_16_feat(train=False, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=True, only_pos=False)


    # val_ds = CAMELYON_16(train=False, transform=transform_data, downsample=1, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    bank = []
    num_bank = 0
    total = 0
    for batch_idx, (_, labels, index) in enumerate(train_loader):
        bag_label = labels[1]
        if bag_label == 0:
            continue
        patch_label = labels[0]# 1,490
        num = 0
        num = torch.sum(patch_label.flatten())
        # for i in range(patch_label.shape[1]):
        #     patch_label[0]
        num_bank = num_bank + num
        total = total + len(patch_label.flatten())
    ratio = num_bank/total
    print(ratio)

    # print(np.mean(bank))
