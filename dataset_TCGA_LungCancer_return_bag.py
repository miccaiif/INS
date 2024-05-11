import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
import glob
from skimage import io
from tqdm import tqdm
import pandas as pd
from random import sample
from sklearn.utils import shuffle

class TCGA_LungCancer_Feat(torch.utils.data.Dataset):
    # @profile
    def __init__(self, train=True, downsample=1.0, return_bag=False, only_pos=False):
        self.train = train
        self.return_bag = return_bag
        bags_csv = './tcga-dataset/tcga-dataset/TCGA.csv'
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path) * 0.8), :]
        test_path = bags_path.iloc[int(len(bags_path) * 0.8):, :]
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        self.only_pos = only_pos
        
        if downsample < 1.0:
            train_path = train_path.iloc[0:int(len(train_path) * downsample), :]
            test_path = test_path.iloc[0:int(len(test_path) * downsample), :]

        self.patch_feat_all = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        if self.train:
            for i in tqdm(range(len(train_path)), desc='loading data'):
                label, feats = get_bag_feats(train_path.iloc[i])
                self.patch_feat_all.append(feats)
                self.patch_corresponding_slide_label.append(np.ones(feats.shape[0]) * label)
                self.patch_corresponding_slide_index.append(np.ones(feats.shape[0]) * i)
                self.patch_corresponding_slide_name.append(np.ones(feats.shape[0]) * i)
        else:
            for i in tqdm(range(len(test_path)), desc='loading data'):
                label, feats = get_bag_feats(test_path.iloc[i])
                self.patch_feat_all.append(feats)
                self.patch_corresponding_slide_label.append(np.ones(feats.shape[0]) * label)
                self.patch_corresponding_slide_index.append(np.ones(feats.shape[0]) * i)
                self.patch_corresponding_slide_name.append(np.ones(feats.shape[0]) * i)

        self.patch_feat_all = np.concatenate(self.patch_feat_all, axis=0).astype(np.float32)
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label).astype(np.long)
        self.patch_corresponding_slide_index =np.concatenate(self.patch_corresponding_slide_index).astype(np.long)
        self.patch_corresponding_slide_name = np.concatenate(self.patch_corresponding_slide_name)

        self.num_patches = self.patch_feat_all.shape[0]
        self.patch_label_all = np.zeros([self.patch_feat_all.shape[0]], dtype=np.long)  # Patch label is not available and set to 0 !
        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\n".format(len(train_path), self.num_patches))
        
        if self.only_pos:
            self.all_patches_pos_slides = []
            self.patch_corresponding_slide_label_pos = []
            self.patch_corresponding_slide_index_pos = []
            self.patch_corresponding_slide_name_pos = []
            self.num_pos_patches = 0
            for i in range(self.num_patches):
                if self.patch_corresponding_slide_label[i] == 1:
                    self.all_patches_pos_slides.append(self.patch_feat_all[i])
                    self.patch_corresponding_slide_label_pos.append(self.patch_corresponding_slide_label[i])
                    self.patch_corresponding_slide_index_pos.append(self.patch_corresponding_slide_index[i])
                    self.patch_corresponding_slide_name_pos.append(self.patch_corresponding_slide_name[i])
                    self.num_pos_patches += 1
            print("[DATA INFO] num_patches is", self.num_pos_patches)
        
        
    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]
            bag = self.patch_feat_all[idx_patch_from_slide_i, :]
            patch_labels = self.patch_label_all[idx_patch_from_slide_i]  # Patch label is not available and set to 0 !
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            # return bag, slide_label
            bag = torch.from_numpy(bag)
            # print(bag.shape)
            return bag, slide_label
            # return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            if self.only_pos:
                each_label = torch.tensor([1, 1])
                patch_image = self.all_patches_pos_slides[index]
                patch_image = torch.from_numpy(patch_image)
                # each_image_w = torch.nn.functional.dropout(patch_image, p=0.2, training=False)
                # each_image_s = torch.nn.functional.dropout(patch_image, p=0.4, training=False)
                patch_label = 1  # [Attention] patch label is unavailable and set to 0
                patch_corresponding_slide_label = self.patch_corresponding_slide_label_pos[index]
                patch_corresponding_slide_index = self.patch_corresponding_slide_index_pos[index]
                patch_corresponding_slide_name = self.patch_corresponding_slide_name_pos[index]
                return patch_image, patch_image, [patch_label, each_label, patch_corresponding_slide_label,
                                                    patch_corresponding_slide_index,
                                                    patch_corresponding_slide_name], 1, index
            else:
                patch_image = self.patch_feat_all[index]
                patch_image = torch.from_numpy(patch_image)
                # each_image_w = torch.nn.functional.dropout(patch_image, p=0.2, training=False)
                # each_image_s = torch.nn.functional.dropout(patch_image, p=0.4, training=False)

                patch_label = 0  # [Attention] patch label is unavailable and set to 0
                patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
                patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
                patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]
                # give label
                if patch_corresponding_slide_label == 0:
                    each_label = torch.tensor([1, 0])
                    point = 0
                else:
                    each_label = torch.tensor([0, 1])
                    point = 1
                # if not self.train:
                return patch_image, patch_image, [patch_label, each_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                        patch_corresponding_slide_name], point, index
                # return each_image_w, each_image_s, [patch_label, each_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                #                         patch_corresponding_slide_name], point, index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        
        elif self.only_pos:
            return self.num_pos_patches
        else:
            return self.num_patches


def get_bag_feats(csv_file_df):
    feats_csv_path = './tcga-dataset/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(1)
    label[0] = csv_file_df.iloc[1]
    return label, feats

def load_TCGA(batch_size=64):
    train_bag_feat = TCGA_LungCancer_Feat(train=True, return_bag=True, downsample=1)
    test_bag_feat = TCGA_LungCancer_Feat(train=False, return_bag=True, downsample=1)
    train_bag_loader = torch.utils.data.DataLoader(train_bag_feat, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_bag_loader = torch.utils.data.DataLoader(test_bag_feat, batch_size=1,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    train_ds_return_true_instance = TCGA_LungCancer_Feat(train=True, return_bag=False, downsample=0.2)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds_return_true_instance,batch_size=64,num_workers=8,shuffle=True,drop_last=True)
    train_ds_return_pos_instance = TCGA_LungCancer_Feat(train=True, return_bag=False, only_pos=True,downsample=0.1)
    train_pos_loader = torch.utils.data.DataLoader(dataset=train_ds_return_pos_instance,batch_size=64,num_workers=8,shuffle=True,drop_last=True)
    train_sampler_pos = torch.utils.data.distributed.DistributedSampler(train_ds_return_pos_instance)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_ds_return_pos_instance,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler_pos is None),
                                                              num_workers=16,
                                                              pin_memory=True,
                                                              sampler=train_sampler_pos,
                                                              drop_last=True)
    train_sampler_cls = torch.utils.data.distributed.DistributedSampler(train_ds_return_true_instance)
    partial_matrix_train_loader_cls = torch.utils.data.DataLoader(dataset=train_ds_return_true_instance,
                                                                  batch_size=batch_size,
                                                                  shuffle=(train_sampler_cls is None),
                                                                  num_workers=16,
                                                                  pin_memory=True,
                                                                  sampler=train_sampler_cls,
                                                                  drop_last=True)
    slide_true_label = train_ds_return_true_instance.patch_corresponding_slide_label
    slide_pos_label = train_ds_return_pos_instance.patch_corresponding_slide_label
    partialY_cls = torch.zeros(len(slide_true_label), 2)
    partialY = torch.zeros(len(slide_true_label), 2)
    partialY_pos = torch.zeros(len(slide_pos_label), 2)
    for i in range(len(slide_pos_label)):
        partialY_pos[i] = torch.tensor([1, 1])
    for i in range(len(slide_true_label)):
            if slide_true_label[i] == 0:
                # print("negative")
                partialY[i] = torch.tensor([1, 0])
                partialY_cls[i] = torch.tensor([1, 0])
            else:
                partialY[i] = torch.tensor([1, 1])
                partialY_cls[i] = torch.tensor([0, 1])
    print('Average candidate num: ', partialY.sum(1).mean())

    return train_pos_loader, partialY, partialY_pos, train_sampler_pos, \
            train_loader, partialY_cls, train_sampler_cls, \
                val_bag_loader, \
            train_bag_loader


if __name__ == '__main__':
    train_ds_feat = TCGA_LungCancer_Feat(train=True, return_bag=True, downsample=1)
    train_loader = torch.utils.data.DataLoader(train_ds_feat, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    for i, batch in enumerate(
            train_loader):  ## (1,N,512)
        print(batch.shape)
    test_ds_feat = TCGA_LungCancer_Feat(train=False, downsample=1.0)
    train_loader = torch.utils.data.DataLoader(train_ds_feat, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_ds_feat, batch_size=1,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_ds_feat, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
   
    for data in tqdm(train_loader, desc='loading'):
        patch_img = data[0]
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")
