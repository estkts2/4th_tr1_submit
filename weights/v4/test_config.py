_base_ = './config.py'
model = dict(
    pretrained = None
)
data = dict(
    test = dict(
        samples_per_gpu = 12,
        ann_file   = '/aichallenge/temp_dir/4th_anno.json',
        img_prefix = '/aichallenge/temp_dir/4th_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1080),
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]
    )
)