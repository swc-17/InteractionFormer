
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from utils.config import load_config_act
from datamodules import WaymoDataModule
from models import InteractionFormer
from utils.log import Logging

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/train.yaml")
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    args = parser.parse_args()
    config = load_config_act(args.config)
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    Data_config = config.Dataset
    if args.pretrain_ckpt == "":
        model = InteractionFormer(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = InteractionFormer(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt, to_cpu=False, logger=logger)
    datamodule = {
        'waymo': WaymoDataModule,
    }[Data_config.dataset](**vars(Data_config))
    trainer_config = config.Trainer
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path,
                                       filename="{epoch:02d}",
                                       monitor='val_minFDE',
                                       every_n_epochs=1,
                                       save_top_k=5,
                                       mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=trainer_config.accelerator, devices=trainer_config.devices,
                         strategy=strategy,
                         accumulate_grad_batches=trainer_config.accumulate_grad_batches,
                         num_nodes=trainer_config.num_nodes,
                         callbacks=[model_checkpoint, lr_monitor],
                         max_epochs=trainer_config.max_epochs)
    if args.ckpt_path == "":
        trainer.fit(model,
                    datamodule)
    else:
        trainer.fit(model,
                    datamodule,
                    ckpt_path=args.ckpt_path)
