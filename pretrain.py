import os
import shutil

import fairseq
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.plugins import *
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_warmup_lr import WarmupLR

from config import args
from trainFrontend.dataset import LRW
from data.utils import collate_fn
from models.V2Vft import V2V


class LRWLightning(pl.LightningDataModule):
    def __init__(self):
        super(LRWLightning, self).__init__()
        self.kwargs = {"num_workers": args["NUM_WORKERS"], "persistent_workers": True if args["NUM_WORKERS"] > 0 else False, "pin_memory": True}

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.trainData = LRW("train", args["LRW_DATA_DIRECTORY"], args["LRW_HDF5_FILE"], args["WORD_TO_INDEX"], args["STEP_SIZE"], True)
            self.valData = LRW("val", args["LRW_DATA_DIRECTORY"], args["LRW_HDF5_FILE"], args["WORD_TO_INDEX"], args["STEP_SIZE"], False)

        if stage == "test" or stage is None:
            self.testData = LRW("test", args["LRW_DATA_DIRECTORY"], args["LRW_HDF5_FILE"], args["WORD_TO_INDEX"], args["STEP_SIZE"], False)

    def train_dataloader(self):
        return DataLoader(self.trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)


def main():
    pl.seed_everything(args["SEED"])
    torch.set_num_threads(args["NUM_CPU_CORE"])
    LRWDataloader = LRWLightning()
    LRWDataloader.setup('fit')
    model = V2V(
        args['dropout_features'], 
        args['frontend'], 
    )

    writer = pl_loggers.TensorBoardLogger(save_dir=args["CODE_DIRECTORY"], name='log', default_hp_metric=False)
    # removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "checkpoints"):
        shutil.rmtree(args["CODE_DIRECTORY"] + "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args["CODE_DIRECTORY"] + "checkpoints/models",
        filename=
        "pretrain-step_{epoch:04d}-loss_{info/valid_loss:.3f}",
        monitor='info/valid_loss',
        every_n_epochs=1,
        every_n_train_steps=0,
        save_top_k=20,
        mode="min",
        auto_insert_metric_name=False,
        save_weights_only=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callback_list = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        gpus=args["GPU_IDS"],
        benchmark=False,
        deterministic=True,
        logger=writer,
        default_root_dir=args["CODE_DIRECTORY"],
        callbacks=callback_list,
        accelerator="ddp",
        #fast_dev_run=True,
        plugins=DDPPlugin(find_unused_parameters=False), #if args["MODAL"] == "VO" else True
    )
    trainer.fit(model, LRWDataloader)
    return


if __name__ == "__main__":
    main()