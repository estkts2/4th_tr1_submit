CUDA_VISIBLE_DEVICES=0,1 \
    tools/dist_test.sh \
    configs/vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py \
    runs_s30/fixed_f010203041042060708_lr0.003_15of30_warm2000_fromvfnetorg/epoch_10.pth  \
    2 --out out.pickle