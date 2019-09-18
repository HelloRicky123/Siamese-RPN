

# Siamese-RPN

This is a PyTorch implementation of SiameseRPN. This project is mainly based on [SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch) and [DaSiamRPN](https://github.com/foolwood/DaSiamRPN).

For more details about siameseRPN please refer to the paper : [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) by Bo Li, Junjie Yan,Wei Wu, Zheng Zhu, Xiaolin Hu.

This repository includes training and tracking codes. 




## Results

This project can get 0.626 AUC on OTB100, and can get better result than the [DaSiamRPN](https://github.com/foolwood/DaSiamRPN) on 46 videos. Test results of 50 trained models on OTB100 are available in the eval_result.json. The best is the 38 epoch.


## Data preparation:

You should first get VID dataset and youtube-bb dataset. This process is a little troublesome. The part of code has not been formatted by now. If any one do this, please give a git pull request.

python bin/create_dataset_ytbid.py --vid-dir /PATH/TO/ILSVRC2015 --ytb-dir /PATH/TO/YT-BB --output-dir /PATH/TO/SAVE_DATA --num_threads 6

The command above will get a dataset, I put the dataset in the baiduyundisk. Use this data to create lmdb.
链接:https://pan.baidu.com/s/1QnQEM_jtc3alX8RyZ3i4-g  密码:myq4

python bin/create_lmdb.py --data-dir /PATH/TO/SAVE_DATA --output-dir /PATH/TO/RESULT.lmdb --num_threads 12

## Traing phase:

python bin/train_siamrpn.py --data_dir /PATH/TO/SAVE_DATA

## Test phase:

Change the data_path first in the test_OTB.py, then run:

python bin/test_OTB.py -ms /PATH/TO/MODEL -v cvpr2013


## Environment:

python version == 3.6.5

pytorch version == 1.0.0

## Model Download:

Pretrained model on Imagenet: https://drive.google.com/drive/folders/1HJOvl_irX3KFbtfj88_FVLtukMI1GTCR

Model with 0.626 AUC: https://pan.baidu.com/s/1vSvTqxaFwgmZdS00U3YIzQ  keyword:v91k

## Reference

[1] Li B , Yan J , Wu W , et al. High Performance Visual Tracking with Siamese Region Proposal Network[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.
