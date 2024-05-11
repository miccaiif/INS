# This file compare features extracted by SSL between CAMELYON16 and Cervical
# Using T-SNE, CAMELYON16 under 5x magnification
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_CAMELYON16_BasedOnFeat import CAMELYON_16_feat
from dataset_CervicalCancer_feat import CervicalCaner_16_feat
# from Datasets_loader.dataset_TCGA_LungCancer import TCGA_LungCancer_Feat
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def t_SNE_vis(feat, label, perplexity=[5,10,20,30,40,50]):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    for pp in perplexity:
        X_tsne = TSNE(n_components=2, random_state=33, perplexity=pp).fit_transform(feat)
        X_tsne_dataset_i = X_tsne[:, :]
        y_dataset_i = label[:]

        plt.figure()
        class_colors = ['darkgray', 'red']
        class_names = ["Neg", "Pos"]
        # for class_i, color_i, name_i in zip(range(3), class_colors, class_names):
        #     print(class_i, color_i, name_i)
        for class_i, color_i, name_i in zip(range(3), class_colors, class_names):
            plt.scatter(X_tsne_dataset_i[y_dataset_i == class_i, 0],
                        X_tsne_dataset_i[y_dataset_i == class_i, 1],
                        c=color_i, label=name_i, s=15, alpha=0.5)
            print(class_i,1)
            if class_i == 0:
                data = pd.DataFrame({'x': X_tsne_dataset_i[y_dataset_i == class_i, 0],'y': X_tsne_dataset_i[y_dataset_i == class_i, 1]})
                data.to_csv("experiment_final/tsne/tsne_simclr.csv", index=True)
            else:
                data = pd.DataFrame({'x': X_tsne_dataset_i[y_dataset_i == class_i, 0],
                                     'y': X_tsne_dataset_i[y_dataset_i == class_i, 1]})
                data.to_csv("experiment_final/tsne/tsne_simclr1.csv", index=True)
        plt.legend()
        # with open('X_tsne_dataset_i.txt', 'a') as f:
        #     f.write(str(auc) + '\n')

    plt.show()
    return 0


if __name__ == '__main__':
    ds_camelyon_train = CAMELYON_16_feat(train=True, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=False)
    ds_camelyon_val = CAMELYON_16_feat(train=False, transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=False)
    camelyon_feat_train = ds_camelyon_train.all_patches
    camelyon_feat_val = ds_camelyon_val.all_patches
    camelyon_label_train = ds_camelyon_train.patch_label
    camelyon_label_val = ds_camelyon_val.patch_label

    ds_cervical_train = CervicalCaner_16_feat(train=True, return_bag=False)
    ds_cervical_val = CervicalCaner_16_feat(train=False, return_bag=False)
    cervical_feat_train = ds_cervical_train.all_patches
    cervical_feat_val = ds_cervical_val.all_patches
    cervical_label_train = ds_cervical_train.patch_corresponding_slide_label
    cervical_label_val = ds_cervical_val.patch_corresponding_slide_label
    #
    # ds_tcga_train = TCGA_LungCancer_Feat(train=True, downsample=1.0, return_bag=False)
    # # ds_tcga_val = TCGA_LungCancer_Feat(train=False, downsample=1.0, return_bag=False)
    # tcga_feat_train = ds_tcga_train.patch_feat_all
    # # tcga_feat_val = ds_tcga_val.all_patches
    # tcga_label_train = ds_tcga_train.patch_label_all
    # # tcga_label_val = ds_tcga_val.patch_label
    train_label_feat = np.load("./results_CAMELYON_feats/train_label_feat_NoDrop.npy")
    test_label_feat = np.load("./results_CAMELYON_feats/test_label_feat_NoDrop.npy")
    # train_label_feat_cc = np.load("./results_CAMELYON_feats_cc/train_label_feat_NoDrop.npy")
    camelyon_label_train = train_label_feat[:, 0:1].astype(np.compat.long).flatten()
    camelyon_feat_train = train_label_feat[:, 1:].astype(np.float32)
    # cc_label_train = train_label_feat_cc[:, 0:1].astype(np.compat.long).flatten()
    # cc_feat_train = train_label_feat_cc[:, 1:].astype(np.float32)
    camelyon_label_val = test_label_feat[:, 0:1].astype(np.compat.long).flatten()
    camelyon_feat_val = test_label_feat[:, 1:].astype(np.float32)
    trans = MinMaxScaler()
    # cc_feat_train = trans.fit_transform(cc_feat_train)
    t_SNE_vis(camelyon_feat_train[::10], camelyon_label_train[::10], perplexity=[30])
    # t_SNE_vis(camelyon_feat_train[::100], np.zeros_like(camelyon_label_train[::100]), perplexity=[5])
    # t_SNE_vis(cervical_feat_train[::10], cervical_label_train[::10], perplexity=[30])
    # t_SNE_vis(cc_feat_train[::300], cc_label_train[::300], perplexity=[50])

    # # t_SNE_vis(tcga_feat_train[::300], tcga_label_train[::300], perplexity=[30])
    #
    # t_SNE_vis(camelyon_feat_train[::10], camelyon_label_train[::10], perplexity=[60])
    # t_SNE_vis(camelyon_feat_train[::10], np.zeros_like(camelyon_label_train[::10]), perplexity=[60])
    #
    # t_SNE_vis(cervical_feat_train[::20], cervical_label_train[::20], perplexity=[60])
    # t_SNE_vis(tcga_feat_train[::300], tcga_label_train[::300], perplexity=[60])

    plt.show()
    print("END")


