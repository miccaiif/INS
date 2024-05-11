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
from utils.randaugment import RandomAugment


def statistics_slide(slide_path_list):
    num_pos_patch_allPosSlide = 0
    num_patch_allPosSlide = 0
    num_neg_patch_allNegSlide = 0
    num_all_slide = len(slide_path_list)

    for i in tqdm(slide_path_list, desc="Statistics"):
        if 'pos' in i.split('/')[-1]:  # pos slide
            num_pos_patch = len(glob.glob(i + "/*_pos*.jpg"))
            num_patch = len(glob.glob(i + "/*.jpg"))
            num_pos_patch_allPosSlide = num_pos_patch_allPosSlide + num_pos_patch
            num_patch_allPosSlide = num_patch_allPosSlide + num_patch
        else:  # neg slide
            num_neg_patch = len(glob.glob(i + "/*.jpg"))
            num_neg_patch_allNegSlide = num_neg_patch_allNegSlide + num_neg_patch

    print("[DATA INFO] {} slides totally".format(num_all_slide))
    print("[DATA INFO] pos_patch_ratio in pos slide: {:.4f}({}/{})".format(
        num_pos_patch_allPosSlide / num_patch_allPosSlide, num_pos_patch_allPosSlide, num_patch_allPosSlide))
    print("[DATA INFO] num of patches: {} ({} from pos slide, {} from neg slide)".format(
        num_patch_allPosSlide+num_neg_patch_allNegSlide, num_patch_allPosSlide, num_neg_patch_allNegSlide))
    return num_patch_allPosSlide + num_neg_patch_allNegSlide


class CAMELYON_16(torch.utils.data.Dataset):
    # @profile
    def __init__(self, root_dir='./patches_byDSMIL',
                 train=True, transform=None, downsample=0.2, drop_threshold=0.0, preload=False, return_bag=False, only_pos=False):
        self.root_dir = root_dir
        self.train = train
        # self.transform = transform
        self.downsample = downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        self.return_bag = return_bag
        self.only_pos = only_pos
        
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x

        # self.strong_transform = transforms.Compose(
        #     [
        #     transforms.ToPILImage(),
        #     transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     RandomAugment(3, 5), ## more than weak_transform
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

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
        # 2.extract all available patches and build corresponding labels
        if self.preload:
            self.all_patches = np.zeros([self.num_patches, 512, 512, 3], dtype=np.uint8)
        else:
            self.all_patches = []
        self.patch_label = []
        # self.all_patches_preload = np.zeros([self.num_patches, 512, 512, 3], dtype=np.uint8)
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            for j in os.listdir(i):
                if self.preload:
                    self.all_patches[cnt_patch, :, :, :] = io.imread(os.path.join(i, j))
                else:
                    self.all_patches.append(os.path.join(i, j))
                self.patch_label.append(int('pos' in j))
                self.patch_corresponding_slide_label.append(int('pos' in i.split('/')[-1]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(i.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        if not self.preload:
            self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
        print(self.all_patches.shape)
        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
            self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))

        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            index_name = self.all_patches[idx_patch_from_slide_i]
            bag = []
            
            if not self.train:
                for idx in index_name:
                    patch = io.imread(idx)
                    bag.append(patch)
                bag = np.array(bag)
                # print(bag.shape)
                bag_normed = np.zeros([bag.shape[0], 3, 512, 512], dtype=np.float32)
                for i in range(bag.shape[0]):
                    bag_normed[i, :, :, :] = self.transform(Image.fromarray(bag[i]))
                bag = bag_normed
            else:
                for idx in index_name:
                    patch = io.imread(idx)
                    patch = self.weak_transform(patch)
                    bag.append(patch)
                    
            patch_labels = self.patch_label[idx_patch_from_slide_i]
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
            if self.preload:
                patch_image = self.all_patches[index]
            else:
                patch_image = io.imread(self.all_patches[index])
            # each_image_w = self.weak_transform(patch_image)
            # each_image_s = self.strong_transform(patch_image)
            # patch_image = torch.from_numpy(patch_image)
            # print('patch_image.shape',patch_image.shape)
            each_image_w = self.weak_transform(patch_image)
            # print(each_image_w.shape)
            each_image_s = self.strong_transform(patch_image)
            patch_image = self.transform(patch_image)
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
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
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            if not self.train:
                return patch_image, patch_image, each_label, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,patch_corresponding_slide_name], point, index
            return each_image_w, each_image_s, each_label, patch_label, point, index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


def cal_img_mean_std():
    train_ds = CAMELYON_16(train=True, transform=None, downsample=1.0, return_bag=False)
    # print(len(train_ds)) # 609655 617
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                               shuffle=False, num_workers=1, drop_last=True, pin_memory=True)
    print(train_loader)
    print("Length of dataset: {}".format(len(train_ds)))
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in tqdm(train_loader, desc="Calculating Mean and Std"):
        # 225报错
        img = data[0]
        for d in range(3):
            mean[d] += img[:, d, :, :].mean()
            std[d] += img[:, d, :, :].std()
    mean.div_(len(train_ds))
    std.div_(len(train_ds))
    mean = list(mean.numpy()*128)
    std = list(std.numpy()*128)
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))
    return mean, std

def load_CAM16_end(batch_size=64):
    transform_data = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x
            # transforms.Normalize(mean=[0.64715815, 0.48541722, 0.65863925], std=[0.24745935, 0.2785922, 0.22133236])])  # CAMELYON16_224x224_5x
    train_ds_return_pos_instance = CAMELYON_16(train=True, transform=None, downsample=0.6, return_bag=False, only_pos=True)
    train_ds_return_true_instance = CAMELYON_16(train=True, transform=None, downsample=0.8, return_bag=False,only_pos=False)
    val_ds_return_instance = CAMELYON_16(train=False, transform=None, downsample=1, return_bag=False)
    val_ds_return_bag = CAMELYON_16(train=False, transform=None, downsample=1, return_bag=True)
    train_ds_return_bag= CAMELYON_16(train=True, transform=None, downsample=0.5, return_bag=True)
    # for images, labels, index in val_ds_return_bag:
    #     print(images.shape)
    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=64, shuffle=False,
                                                      num_workers=8, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False,
                                                      num_workers=8, drop_last=False)
    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=False,
                                                      num_workers=0, drop_last=False)
    # for batch_idx, (images_w, images_s, labels, index) in enumerate(val_loader_instance):
    #     print(batch_idx)
    train_sampler_pos = torch.utils.data.distributed.DistributedSampler(train_ds_return_pos_instance)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_ds_return_pos_instance,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler_pos is None),
                                                              num_workers=16,
                                                              pin_memory=True,
                                                              sampler=train_sampler_pos,
                                                              drop_last=True)
    # return train_loader, train_givenY, train_sampler, test_loader
    train_sampler_cls = torch.utils.data.distributed.DistributedSampler(train_ds_return_true_instance)
    partial_matrix_train_loader_cls = torch.utils.data.DataLoader(dataset=train_ds_return_true_instance,
                                                                  batch_size=batch_size,
                                                                  shuffle=(train_sampler_cls is None),
                                                                  num_workers=16,
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
            val_loader_bag, train_loader_bag
            #,\
            #train_ds_return_true_instance.patch_label, train_ds_return_pos_instance.patch_label, train_loader_bag

if __name__ == '__main__':
    # mean, std = cal_img_mean_std() #
    # load_CAM16_end()
    transform_data = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x
            # transforms.Normalize(mean=[0.64715815, 0.48541722, 0.65863925], std=[0.24745935, 0.2785922, 0.22133236])])  # CAMELYON16_224x224_5x
    train_ds = CAMELYON_16(train=True, transform=transform_data, downsample=1, return_bag=True)


    # val_ds = CAMELYON_16(train=False, transform=transform_data, downsample=1, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    bank = []
    for batch_idx, (images, labels, index) in enumerate(train_loader):
        bag_label = labels[1]
        if bag_label == 0:
            continue
        patch_label = labels[0]# 1,490
        num = 0
        num = torch.sum(patch_label.flatten())
        # for i in range(patch_label.shape[1]):
        #     patch_label[0]
        ratio = (float)(num) / patch_label.shape[1]
        bank.append(ratio)
        print(batch_idx,' ', ratio)

    print()

    # val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64,
    #                                          shuffle=False, num_workers=8, drop_last=False, pin_memory=True)
    # for data in train_loader:
    #     patch_img = data[0]
    #     label_patch = data[1][0]
    #     label_bag = data[1][1]
    #     idx = data[-1]
    # print("END")
