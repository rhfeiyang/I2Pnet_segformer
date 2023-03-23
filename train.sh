cd ~/code/I2Pnet_segformer
python train_coarse.py models/iter_mask/coarse_segformer_cocolvis_itermask_3p.py --workers 4 --exp-name segformer_coarse --gpus 0 && python train_fine.py models/iter_mask/fine_segformer_cocolvis_itermask_3p.py --workers 4 --int-model segformer_corase --exp-name segformer_fine --gpus 0
