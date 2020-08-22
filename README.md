## Strip Pooling: Rethinking Spatial Pooling for Scene Parsing
##### Yongq
in Haiyang Liu, Yichen Wang, Jiayi Zhao, Guowu Yang, and Fengmao Lv.
##### Abstract:
> <sup> Semantic segmentation, which aims to acquire a detailed understanding of images, is an essential issue in computervision. However, in practical scenarios, new categories that aredifferent from the categories in training usually appear. Since itis impractical to collect labeled data for all categories, how toconduct zero-shot learning in semantic segmentation establishesan important problem. Although the attribute embedding ofcategories can promote effective knowledge transfer across different categories, the prediction of segmentation network revealsobvious bias to seen categories. In this paper, we propose an easyto-implement transductive approach to alleviate the predictionbias in zero-shot semantic segmentation. Our method assumesthat both the source images with full pixel-level labels andunlabeled target images are available during training. To bespecific, the source images are used to learn the relationshipbetween visual images and semantic embeddings, while the targetimages are used to alleviate the prediction bias towards seencategories. We conduct comprehensive experiments on diversesplit s of the PASCAL dataset. The experimental results clearlydemonstrate the effectiveness of our method.</sup>

This repository is a PyTorch implementation for our [paper](https://arxiv.org/pdf/2007.00515.pdf) (non-commercial use only).

##### Datasets:
Please download pascal-VOC and place the data in the corresponding path of 'DATA_PATH'.

##### Initialization Model:
Use pytorch's built-in model pre-trained through imagenet

##### Train:
First of all, train through train_split_lambda.py. 
Then use train_st_split.py to selftrain the results of the previous step.

##### Hyperparameters:
Introduce some important parameter settings:
```
split : change the partition of the dataset
RESTORE_FROM_WHERE : change the model training mode
    "pretrained" : Use the pre-training model
    "continue" : continue to train the model
    "saved" : selftrain training for the first time (for train_st_split.py)
```

##### Citation:
If you find this useful, please cite our work as follows:
```
@misc{liu2020learning,
    title={Learning unbiased zero-shot semantic segmentation networks via transductive transfer},
    author={Haiyang Liu and Yichen Wang and Jiayi Zhao and Guowu Yang and Fengmao Lv},
    year={2020},
    eprint={2007.00515},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```