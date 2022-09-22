_base_ = ['./faster_rcnn_r50_fpn_1x_duor.py']

data_root = 'data/duo_glnet/'

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