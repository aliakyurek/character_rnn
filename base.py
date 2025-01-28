"""
This is the Pytorch Lightning (PL) application base module. It contains the PL application class named PlApp
to glue the PL data module, model, and experiment classes together.
It arranges checkpointing, checkpoint resuming, logging, tensorboarding, and training.
"""

import os
import yaml
import re
import torch
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.utilities import rank_zero_only


class PlApp:
    class TBLogger(loggers.TensorBoardLogger):
        """
        This is a custom TensorBoardLogger class to remove the 'epoch' key from the metrics dictionary.
        Because it looks awkward to have 'epoch' as metric in the TensorBoard.
        """
        @rank_zero_only
        def log_metrics(self, metrics, step):
            metrics.pop('epoch', None)
            return super().log_metrics(metrics, step)
    
    @staticmethod
    def init_env(exp_path):
        """
        This is a static method to read the configuration file and initialize the environment resetting the random seeds
        """
        with open(f"{exp_path}/params.yml",'r') as file:
            params = yaml.safe_load(file)
        pl.seed_everything(params['app']['manual_seed'])
        torch.cuda.empty_cache()
        return params
    
    def __init__(self, params, data_module, model, experiment, ckpt_path=None):
        """
        This is the constructor of the PL application class.
        It handles logging, metric monitoring, save/load checkpoints, and training.
        :param params: The configuration dictionary.
        :param data_module: The PL data module instance.
        :param model: The Pytorch model instance.
        :param experiment: The PL experiment class.
        :param ckpt_path: The checkpoint file path to load and continue training (optional).
        """
        self.ckpt_path = ckpt_path
        self.data_module = data_module
        p=params['app']
        version=None

        # if ckpt_path is provided, load the experiment from the checkpoint file.
        # a checkpoint file saved by PL contains a 'state_dict' of model weights, optimizer states, epoch, and global step etc...
        # PL automatically saves found nn.Module instances into the checkpoint file together with the other information.

        # model weights in the 'state_dict' dictionary of the checkpoint file have keys with their full instance path relative to itself like 'model.rnn.weight_ih',
        # model.fc.weight' etc... The values are corresponding weight tensors.
        if self.ckpt_path:
            # load_from_checkpoint works by first instantiating its beloning class using cls and passed kw_args after ckpt_path. These are class' constructor parameters.
            # after that, the parameters of model instance passed to the constructor are replaced with the weights resulting from 
            # the load_from_checkpoint method. Only the values of objects like model are found in state_dict. 
            # For example the lr parameter passed to the experiment is not found in the state_dict, so it's not restored. 
            # Since PL rejects to call load_from_checkpoint method from an instance (although this is technically possible),
            # we access the class from instance and call the method from there.
            self.experiment = experiment.__class__.load_from_checkpoint(self.ckpt_path, model=model, **params['experiment'])

            # extract the version number from the checkpoint file name.
            version = int(re.search(r"version_(.*?)-",self.ckpt_path).group(1))
        else:
            self.experiment = experiment

        # initialize custom tensorboard logger. If a version number is provided, it will be used in the end of log directory name like version_XYZ
        # otherwise, the logger will automatically assign a version number, by finding the next available version number in the log directory.
        logger = self.TBLogger(save_dir=p['logs_dir'], name=p['name'], default_hp_metric=False, version=version)

        callbacks = None
        # if there's a monitor parameter in the trainer section of the configuration, add a ModelCheckpoint callback to save the best model.        
        if('monitor' in params['trainer']):
            callbacks = [pl.callbacks.ModelCheckpoint(
                monitor=params['trainer']['monitor'], mode=params['trainer']['mode'],
                dirpath=os.path.join(p['logs_dir'],p['name']), save_top_k=1, save_weights_only=False,
                filename=f"version_{logger.version:02d}-" + "{epoch:02d}-{"+params['trainer']['monitor']+":.3f}"
            )]

        # initialize the PL trainer instance
        self.trainer = pl.Trainer(accelerator="auto", max_time=params['trainer']['max_time'],
                                  benchmark=False, deterministic=True, num_sanity_val_steps=0,
                                  enable_model_summary=False, enable_progress_bar=True,
                                  logger=logger, callbacks=callbacks,
                                  limit_train_batches=None, limit_val_batches=None, limit_test_batches=None,
                                  gradient_clip_val=1.0)

        # log hyperparameters (these are actually values from the configuration file) to the log directory.
        self.trainer.logger.log_hyperparams(params)

    def train(self):
        # even though we load the checkpoint in constructor, load_from_checkpoint is intended to be used for inference, so 
        # optimizer states, epoch, and global step are not restored to experiment instance.
        # if we want to continue training of restored checkpoint, we should call fit method of the trainer instance.
        # trainer.fit restores the whole training progress.
        # It also restores the model weights causing a double work but it's OK.
        return self.trainer.fit(self.experiment, datamodule=self.data_module, ckpt_path=self.ckpt_path)

