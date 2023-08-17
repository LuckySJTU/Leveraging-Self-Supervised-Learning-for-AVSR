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
from data.lrs2_dataset import LRS2
from data.utils import collate_fn
from models.V2Vft import VisFeatureExtractionModel
from models.LAS import att_deocder
from utils.decoders import ctc_greedy_decode, teacher_forcing_attention_decode
from utils.label_smoothing import SmoothCTCLoss, SmoothCrossEntropyLoss
from utils.metrics import compute_error_ch, compute_error_word


class LRS2Lightning(pl.LightningDataModule):
    def __init__(self):
        super(LRS2Lightning, self).__init__()
        self.kwargs = {"num_workers": args["NUM_WORKERS"], "persistent_workers": True if args["NUM_WORKERS"] > 0 else False, "pin_memory": True}

    def setup(self, stage):
        if stage == "fit" or stage is None:
            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": args["NOISE_PROBABILITY"], "noiseSNR": args["NOISE_SNR_DB"]}
            self.trainData = LRS2(args['MODAL'], "train", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                                  True, noiseParams)

            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}
            self.valData = LRS2(args['MODAL'], "val", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False,
                                noiseParams)

        if stage == "test" or stage is None:
            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}
            self.testData = LRS2(args['MODAL'], "test", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False,
                                 noiseParams)

    def train_dataloader(self):
        return DataLoader(self.trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)


class V2V(pl.LightningModule):
    def __init__(self, dropout_features, frontend, output_size):
        super(V2V, self).__init__()
        self.feature_extractor = VisFeatureExtractionModel(frontend)
        self.dropout_feats = nn.Dropout(p=dropout_features)

        self.vocab_size = output_size
        
        hidden_dim = {
            "resnet18":512,
            "resnet50":2048,
        }
        out_dim = hidden_dim[frontend] #self.feature_extractor.MoCoModel[-1][1].out_channels
        # self.biGRU = nn.GRU(input_size, 512, 2, batch_first=True, bidirectional=True)
        # self.dropout = nn.Dropout(p=0.25)

        # out_dim = 512
        # self.vsr_proj = nn.Linear(out_dim, output_size)

        # self.vsr_bert = AutoModelForTokenClassification.from_pretrained("bert-base-uncased",num_labels=output_size)
        # self.vsr_bert.bert.embeddings = nn.Identity()

        self.att_deocder = att_deocder(
            encode_size=out_dim,
            dec_dim=512,
            att_dim=1024,
            vocab_size=output_size,
            init_adadelta=True,
            ctc_weight=0.5,
            attention="dot",
            decoder="LSTM",
            emb_drop=0,
        )


    def forward(self, source, sizes, targetinBatch, targetoutBatch, target_lengths):
        frames = source.shape[1]
        if self.avhubert:
            x=self.backbone.forward_features(source.permute(0,2,1,3,4),'video')
            # x=x.permute(0,2,1)
        else:
            features = self.backbone.feature_extractor(source)  # (B*N) x D
            features = features.view(-1, frames, features.shape[-1]).permute(0, 2, 1)     # B x D x N
            x = self.backbone.dropout_feats(features)
            # x = self.backbone.feature_aggregator(x)     # B x D x N
            # x = self.backbone.dropout_agg(x)
        cls = None
        vsr = None
        att = None
        if self.wb and word_boundary:
            word_boundary = torch.tensor(word_boundary,device=x.device)
            word_boundary = word_boundary.unsqueeze(2)
            x = x.permute(0,2,1)
            x = torch.dstack((x,word_boundary))
            x = x.permute(0,2,1)

        if self.LRW:
            if self.cls_type == "TCN":
                cls = self.tcn(x)
                cls = torch.mean(cls, -1)
                cls = self.cls_proj(cls)
            elif self.cls_type=="Linear":
                cls = self.cls_proj(torch.mean(x,-1))
            elif self.cls_type=="MSTCN":
                cls = self.mstcn(x)
                cls = sum(cls)
                cls = torch.mean(cls,-1)
                cls = self.cls_proj(cls)
            else:
                # default GRU
                h, _ = self.gru(x.permute(0, 2, 1))
                cls = self.cls_proj(self.dropout(h)).mean(1)
        
        if self.vsr:
            # vsr, _ = self.biGRU(x.permute(0, 2, 1))
            # vsr = self.dropout(vsr)
            # vsr = self.vsr_proj(vsr)

            # vsr = self.vsr_proj(x.permute(0,2,1))

            # vsr = self.vsr_bert(inputs_embeds=x.permute(0,2,1)) # batch*512*length
            # vsr = vsr.logits

            teacher_forcing = 1 if targets is not None else 0
            teach = None
            if teacher_forcing:
                # teach: <bos>Sentence<eos>, but only use <bos>Sentence
                # target: Sentence<eos>
                teach = torch.zeros((targets.shape[0],targets.shape[1]+1,self.vocab_size),dtype=torch.long,device=targets.device)
                for i in range(targets.shape[0]):
                    teach[i,1:,:] = F.one_hot(targets[i].long(), num_classes=self.vocab_size)
                teach[:,0,0] = 1
            
            vsr, att, att_seq, dec_state = self.att_deocder(
                x.permute(0,2,1), 
                torch.tensor(sizes,device=x.device), 
                max(target_lengths), 
                tf_rate=teacher_forcing, 
                teacher=teach,
            )
            # att = self.att_proj(att)

            # encoder_out = {"encoder_out": [x.permute(2,0,1)], "encoder_padding_mask": padding_mask}
            # att, otherArgs = self.lm_decoder(targets, encoder_out=encoder_out)


        return vsr, cls, att


    def training_step(self, batch, batch_idx):
        inputBatch, targetinBatch, targetoutBatch, targetLenBatch = batch
        Alpha = self.trainParams["Alpha"]

        if self.trainParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.trainParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
        targetinBatch = targetinBatch.int()
        targetoutBatch = targetoutBatch.int()
        targetLenBatch = targetLenBatch.int()
        targetMask = torch.zeros_like(targetoutBatch, device=self.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]

        inputLenBatch, outputBatch = self(inputBatch, targetinBatch, targetLenBatch.long(), True)
        with torch.backends.cudnn.flags(enabled=False):
            ctcloss = self.CTCLossFunction[0](outputBatch[0], concatTargetoutBatch, inputLenBatch, targetLenBatch)
            celoss = self.CELossFunction[0](outputBatch[1], targetoutBatch.long())
            loss = Alpha * ctcloss + (1 - Alpha) * celoss
        self.log("info/train_ctcloss", ctcloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/train_celoss", celoss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0].detach(), inputLenBatch, self.trainParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/train_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.trainParams["spaceIx"])
        self.log("info/train_WER", w_edits / w_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputBatch, targetinBatch, targetoutBatch, targetLenBatch = batch
        Alpha = self.trainParams["Alpha"]

        if self.valParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.valParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
        targetinBatch = targetinBatch.int()
        targetoutBatch = targetoutBatch.int()
        targetLenBatch = targetLenBatch.int()
        targetMask = torch.zeros_like(targetoutBatch, device=self.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]

        # inputBatch: b*f*1*112*112; targetinBatch: b*L; targetLenBatch: b
        inputLenBatch, outputBatch = self(inputBatch, targetinBatch, targetLenBatch.long(), False)
        with torch.backends.cudnn.flags(enabled=False):
            ctcloss = self.CTCLossFunction[0](outputBatch[0], concatTargetoutBatch, inputLenBatch, targetLenBatch)
            celoss = self.CELossFunction[0](outputBatch[1], targetoutBatch.long())
            loss = Alpha * ctcloss + (1 - Alpha) * celoss
        self.log("info/val_ctcloss", ctcloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/val_celoss", celoss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("info/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0], inputLenBatch, self.valParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/val_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.valParams["spaceIx"])
        self.log("info/val_WER", w_edits / w_count, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        predictionBatch, predictionLenBatch = teacher_forcing_attention_decode(outputBatch[1], self.valParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/val_TF_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.valParams["spaceIx"])
        self.log("info/val_TF_WER", w_edits / w_count, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
        scheduler_reduce = ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"], patience=args["LR_SCHEDULER_WAIT"],
                                             threshold=args["LR_SCHEDULER_THRESH"], threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
        if args["LRW_WARMUP_PERIOD"] > 0:
            scheduler = WarmupLR(scheduler_reduce, init_lr=args["FINAL_LR"], num_warmup=args["LRS2_WARMUP_PERIOD"], warmup_strategy='cos')
            scheduler.step(1)
        else:
            scheduler = scheduler_reduce

        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'epoch',  # The unit of the scheduler's step size
                'frequency': 1,  # The frequency of the scheduler
                'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
                'monitor': 'CER/val_CER',
                'strict': True,  # Whether to crash the training if `monitor` is not found
                'name': None,  # Custom name for LearningRateMonitor to use
            }
        }
        return optim_dict


def main():
    pl.seed_everything(args["SEED"])
    torch.set_num_threads(args["NUM_CPU_CORE"])
    LRS2Dataloader = LRS2Lightning()
    LRS2Dataloader.setup('fit')
    modelargs = (args["DMODEL"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"], args["AUDIO_FEATURE_SIZE"],
                 args["VIDEO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["CHAR_NUM_CLASSES"])
    model = V2V(0.1, "resnet18", 40)
    # loading the pretrained weights
    if not args["MODAL"] == "AV" and args["TRAIN_LRS2_MODEL_FILE"] is not None:
        stateDict = torch.load(args["TRAIN_LRS2_MODEL_FILE"], map_location="cpu")['state_dict']
        model.load_state_dict(stateDict, strict=False)
    if args["MODAL"] == "AV" and args["TRAINED_AO_FILE"] is not None and args["TRAINED_VO_FILE"] is not None:
        AOstateDict = torch.load(args["TRAINED_AO_FILE"])['state_dict']
        stateDict = torch.load(args["TRAINED_VO_FILE"])['state_dict']
        for k in list(AOstateDict.keys()):
            if not (k.startswith('audioConv') or k.startswith('wav2vecModel')):
                del AOstateDict[k]

        for k in list(stateDict.keys()):
            if not (k.startswith('videoConv') or k.startswith('visualModel')):
                del stateDict[k]
        stateDict.update(AOstateDict)
        model.load_state_dict(stateDict, strict=False)

    writer = pl_loggers.TensorBoardLogger(save_dir=args["CODE_DIRECTORY"], name='log', default_hp_metric=False)
    # removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "checkpoints"):
        shutil.rmtree(args["CODE_DIRECTORY"] + "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args["CODE_DIRECTORY"] + "checkpoints/models",
        filename=
        "train-step_{epoch:04d}-cer_{CER/val_CER:.3f}" if args["LR_SCHEDULER_METRICS"] == "CER" else "train-step_{epoch:04d}-wer_{info/val_WER:.3f}",
        monitor='CER/val_CER' if args["LR_SCHEDULER_METRICS"] == "CER" else 'info/val_WER',
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
        accelerator="dp",
        #plugins=DDPPlugin(find_unused_parameters=False if args["MODAL"] == "VO" else True),
    )
    trainer.fit(model, LRS2Dataloader)
    return


if __name__ == "__main__":
    main()
