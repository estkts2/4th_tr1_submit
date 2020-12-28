_base_ = './vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
model = dict(
    #pretrained='open-mmlab://res2net101_v1d_26w_4s',
    pretrained = None,
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


###########################################################################

#albu_train_transforms = [
#    dict(
#        type='RandomBrightnessContrast',
#        #brightness_limit=[0.1, 0.3],
#        brightness_limit=[0.1, 0.15],
#        #contrast_limit=[0.1, 0.3],
#        contrast_limit=[0.1, 0.15],
#        p=0.2),
#    dict(
#        type='OneOf',
#        transforms=[
#            dict(
#                type='RGBShift',
#                #r_shift_limit=10,
#                #g_shift_limit=10,
#                #b_shift_limit=10,
#                #p=1.0),
#                r_shift_limit=3,
#                g_shift_limit=3,
#                b_shift_limit=3,
#                p=0.7),
#            dict(
#                type='HueSaturationValue',
#                #hue_shift_limit=20,
#                #sat_shift_limit=30,
#                #val_shift_limit=20,
#                #p=1.0)
#                hue_shift_limit=10,
#                sat_shift_limit=10,
#                val_shift_limit=5,
#                p=0.7)
#        ],
#        #p=0.1),
#        p=0.7),
#    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
#    dict(type='ChannelShuffle', p=0.1),
#    dict(
#        type='OneOf',
#        transforms=[
#            dict(type='Blur', blur_limit=3, p=1.0),
#            dict(type='MedianBlur', blur_limit=3, p=1.0)
#        ],
#        p=0.1),
#]

###########################################################################
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        #img_scale=[(1333, 480), (1333, 960)],
        img_scale=[(1920, 1080), (1333, 960)],
        #img_scale=(960, 540),
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    
    #dict(
    #    type='Albu',
    #    transforms=albu_train_transforms,
    #    bbox_params=dict(
    #        type='BboxParams',
    #        format='pascal_voc',
    #        label_fields=['gt_labels'],
    #        min_visibility=0.0,
    #        filter_lost_elements=True),
    #    keymap={
    #        'img': 'image',
    #        'gt_masks': 'masks',
    #        'gt_bboxes': 'bboxes'
    #    },
    #    update_pad_shape=False,
    #    skip_img_without_anno=True),
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        #img_scale=[(1200, 800), (1300, 900), (1500, 1000), (1600, 1200)],
        #img_scale=[(1920, 1080), (1280,720) (960, 540)],
        img_scale=(1920, 1080),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline))

###########################################################################

#optimizer = dict(_delete_=True, type='Adam', lr=0.0003, weight_decay=0.0001)
#optimizer = dict(_delete_=True, type='Adam', lr=0.00075, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.00075, momentum=0.9, weight_decay=0.0001)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[8])
total_epochs = 30 


#load_from = 'weights/vfnet_r2_101_dcn_ms_2x_51.1.pth' 
load_from =  'runs_iitp_1fps_16fp/540p_lr0.00075_8of30_warm1000_loadfrom8ep/epoch_4.pth'

#load_from = 'runs/f0f1f2f3f41f42_sgd_0.00075_warm_1000/epoch_14.pth' 
#load_from = 'runs_15fps/f0f1f2f3f41f42f60_caug_neg0.2_lr0.0005_20of30_warm500_scratch/epoch_19.pth' 
#resume_from = 'runs_1fps/resized_mobileval_lr0.003_16of30_warm1000_loadfromorg/epoch_15.pth'
workflow = [('train', 1)]#, ('test', 1)]

fp16 = dict(loss_scale=512.)
