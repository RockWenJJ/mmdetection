import os.path as osp
import mmcv
import numpy as np
import torch
from collections import OrderedDict
from mmcv.runner.dist_utils import master_only
from mmcv.runner import HOOKS
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook


@HOOKS.register_module()
class UIEWandbLoggerHook(WandbLoggerHook):
    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 vis_interval=1000,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 **kwargs):
        super().__init__(init_kwargs, interval, **kwargs)
        
        self.vis_interval = vis_interval
        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = log_checkpoint_metadata
        self.ckpt_hook: CheckpointHook = None
    
    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
        
        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log checkpoint in MMDetWandbHook, `CheckpointHook` is'
                    'required, please check hooks in the runner.')
            else:
                self.ckpt_interval = self.ckpt_hook.interval  # by epoch
    
    @master_only
    def after_train_iter(self, runner):
        # super().after_train_iter(runner)
        
        # log losses
        if self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
            
            if runner.log_buffer.ready:
                log_dict = self.log(runner)
                self.wandb.log(log_dict, step=self.get_iter(runner))
        
        # log images
        mode = self.get_mode(runner)
        if self.every_n_iters(runner, self.vis_interval):
            img_dict = {f'{mode}/predicts': runner.outputs['images']['predict'].cpu().data,
                        f'{mode}/targets': runner.outputs['images']['target'].cpu().data,
                        f'{mode}/input': runner.outputs['images']['input'].cpu().data}
            for k, v in img_dict.items():
                img_dict[k] = self.wandb.Image(v)
            self.wandb.log(img_dict, step=self.get_iter(runner))
        
        # log checkpoints
        if self.reset_flag:
            runner.log_buffer.clear_output()
    
    @master_only
    def after_train_epoch(self, runner):
        # save ckpt as artifacts
        if (self.log_checkpoint and self.every_n_epochs(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_iter(runner))):
            if self.log_checkpoint_metadata:
                metadata = {'iter': runner.iter + 1}
            else:
                metadata = None
            aliases = [f'iter_{runner.iter + 1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'epoch_{runner.epoch + 1}.pth')  # the saved pth should be the same with ckpt_hook's pth
            # self._log_ckpt_as_artifact(model_path, aliases, metadata)
        
        # log metrics
        if runner.log_buffer.ready:
            log_dict = self.log(runner)
            self.wandb.log(log_dict, step=self.get_iter(runner))
            runner.log_buffer.clear_output()
    
    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=False)
        
        mode = self.get_mode(runner)
        log_dict = OrderedDict(
            epoch=self.get_epoch(runner),
            iter=cur_iter)
        log_dict[f'{mode}/epoch'] = self.get_epoch(runner)
        log_dict[f'{mode}/iter'] = cur_iter
        
        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict[f'{mode}/lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict[f'{mode}/lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict[f'{mode}/lr'].update({k: lr_[0]})
        
        for k, v in runner.log_buffer.output.items():
            log_dict[f'{mode}/{k}'] = v
        
        return log_dict
    
    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)