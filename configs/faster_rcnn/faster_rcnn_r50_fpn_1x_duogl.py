_base_ = ['./faster_rcnn_r50_fpn_1x_duor.py']

data_root = 'data/duo_glnet/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + '/images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline))

log_config = dict(
    hooks=[
        dict(type='MMDetWandbHook',
             interval=50,
             init_kwargs={'project': "DUO-Detection", # Project name in WandB
                          'name': 'faster_rcnn_r50_fpn_1x_duoglnet'},
             log_checkpoint=True,
             num_eval_images=100,
             bbox_score_thr=0.3,
             ),
    ])