# Siamese-RPN

This is the code get the 0.5967 auc on the OTB100.

To train the code :

CUDA_VISIBLE_DEVICES=0,1 python bin/train_siamrpn.py --data_dir /dataset_ssd/ytb_vid_rpn_id

To test the code :

CUDA_VISIBLE_DEVICES=2 python bin/test_OTB.py -ms ./models/siamrpn_44.pth -v OTB100

The best epoch is at 50 epoch.

