# :whale: Official-PyTorch-Implementation-of-INS
This is a PyTorch/GPU implementation of our IEEE TCSVT paper: Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Good Instance Classifier is All You Need ([Paper](https://arxiv.org/abs/2307.02249)).

## Abstract
Weakly supervised whole slide image classification is usually formulated as a multiple instance learning (MIL) problem, where each slide is treated as a bag, and the patches cut out of it are treated as instances. Existing methods either train an instance classifier through pseudo-labeling or aggregate instance features into a bag feature through attention mechanisms and then train a bag classifier, where the attention scores can be used for instance-level classification. However, the pseudo instance labels constructed by the former usually contain a lot of noise, and the attention scores constructed by the latter are not accurate enough, both of which affect their performance. In this paper, we propose an instance-level MIL framework based on contrastive learning and prototype learning to effectively accomplish both instance classification and bag classification tasks. To this end, we propose an instance-level weakly supervised contrastive learning algorithm for the first time under the MIL setting to effectively learn instance feature representation. We also propose an accurate pseudo label generation method through prototype learning. We then develop a joint training strategy for weakly supervised contrastive learning, prototype learning, and instance classifier training. Extensive experiments and visualizations on four datasets demonstrate the powerful performance of our method.

<p align="center">
  <img src="https://github.com/miccaiif/INS/blob/main/Figure1.png" width="720">
</p>

## For training and testing
* For a quick start, please run 
```shell
python train_withbag_cam.py --multiprocessing-distributed --wl 0.5
```
Support shell-based training and testing: run [shell_cam.sh](https://github.com/miccaiif/INS/blob/main/shell_cam.sh).

<p align="center">
  <img src="https://github.com/miccaiif/INS/blob/main/Figure3.png" width="720">
</p>

## Data preprocessing
The instance classifier could be trained in an end2end manner with the input of original RGB images or be trained on the basis of pre-trained self-supervised (SSL) feature extractors.
The input data should be pre-formatted to fit the end2end manner [dataset.py](https://github.com/miccaiif/INS/blob/main/dataset_CAMELYON16.py) or SSL manner [dataset.py](https://github.com/miccaiif/INS/blob/main/dataset_CAMELYON16_BasedOnFeat.py).
More details could be referred to the following helpful works:
[DGMIL](https://github.com/miccaiif/DGMIL)
[WENO](https://github.com/miccaiif/WENO)
[DSMIL](https://github.com/binli123/dsmil-wsi)
[CLAM](https://github.com/mahmoodlab/CLAM).

## Acknowledgement
We sincerely thank [PiCO](https://github.com/hbzju/PiCO) for their inspiration and contributions to the codes.

### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).

