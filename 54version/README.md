# Siamese-RPN

This is the code get the 0.54 auc on the OTB50.

To train the code :

CUDA_VISIBLE_DEVICES=2 python bin/train_siamfc.py --data_dir /dataset_ssd/vid15rpn_large --model_path ./siamrpn_30.pth --init

To test the code :

CUDA_VISIBLE_DEVICES=2 python bin/test_OTB.py -ms ./models/siamrpn_44.pth -v cvpr2013

The best epoch is about 55.

The ./siamrpn_30.pth is got after 20 epochs' 8e-4 lr and 30 epochs' 1e-2. 

链接:https://pan.baidu.com/s/1_syPIBqoH7cChdN3cNInfw  密码:m1kh