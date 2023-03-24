cd ~/code/I2Pnet_segformer
python train_fine.py models/iter_mask/fine_segformer_cocolvis_itermask_3p.py --workers 4 --int-model 000_segformer_coarse --exp-name segformer_fine --gpus 0