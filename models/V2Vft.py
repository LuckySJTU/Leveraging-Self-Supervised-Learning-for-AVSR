import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision.models as models
import math
from models.utils import *
from utils.label_smoothing import SmoothCrossEntropyLoss, SmoothCTCLoss
from utils.decoders import compute_CTC_prob
import numpy as np
from tqdm import tqdm

class ZeroPad3D(nn.Module):
    def __init__(self,leftpad,rightpad):
        super().__init__()
        self.leftpad=leftpad
        self.rightpad=rightpad
    def forward(self,x):
        return F.pad(x,(0,0,0,0,self.leftpad,self.rightpad),"constant",0)
    
class ReplicationPad3D(nn.Module):
    def __init__(self,leftpad,rightpad):
        super().__init__()
        self.leftpad=leftpad
        self.rightpad=rightpad
    def forward(self,x):
        return F.pad(x,(0,0,0,0,self.leftpad,self.rightpad),"replicate")
    
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod

class Conv3DAggegator(nn.Module):
    def __init__(
            self,
            conv_layers,
            embed,
            dropout,
            skip_connections,
            residual_scale,
            non_affine_group_norm,
            conv_bias,
            zero_pad,
            activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k[0] // 2
            kb = ka - 1 if k[0] % 2 == 0 else ka

            if zero_pad:
                pad = ZeroPad3D(ka+kb,0)
            else:
                pad = ReplicationPad3D(ka+kb,0)

            return nn.Sequential(
                pad,
                nn.Conv3d(n_in, n_out, k, stride=stride, bias=conv_bias),
                # nn.Dropout(p=dropout),
                # norm_block(True, n_out, affine=not non_affine_group_norm),
                # activation,
                nn.BatchNorm3d(n_out),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv3d(in_d, dim, k, stride, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x): #16*1*29*112*112
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x

class VisFeatureExtractionModel(nn.Module):
    def __init__(self, frontend):
        super(VisFeatureExtractionModel, self).__init__()
        # Conv3D
        self.conv3D = Conv3DAggegator(
            conv_layers=[(64,(3,4,4),(1,2,2)),(64,(3,2,2),(1,2,2)),(64,(1,2,2),(1,1,1))],
            embed = 1,
            dropout=0,
            skip_connections=False,
            residual_scale=0.5,
            non_affine_group_norm=False,
            conv_bias=True,
            zero_pad=True,
            activation=nn.ReLU()
        )
        MoCoModel = models.__dict__[frontend]()
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()
        self.MoCoModel = MoCoModel

        # print(self.MoCoModel)

    def forward(self, x, x_len):
        #input x : 16*29*1*112*112
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3D(x) # 16*64*29*26*26
        x = x.permute(0, 2, 1, 3, 4) #16*29*64*26*26
        # x: B x N x C x H x W
        # x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) #464*64*26*26

        mask = torch.zeros(x.shape[:2], device=x.device)
        mask[(torch.arange(mask.shape[0], device=x.device), x_len - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        x = x[~mask]

        x = self.MoCoModel(x) #464*2048
        return x
        # return shape: (B*N) x 512

        # refer models/moco_visual_frontend.py

class conv1dLayers(nn.Module):
    def __init__(self, MaskedNormLayer, inD, dModel, outD, downsample=False):
        super(conv1dLayers, self).__init__()
        if downsample:
            kernel_stride = 2
        else:
            kernel_stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(inD, dModel, kernel_size=(kernel_stride,), stride=(kernel_stride,), padding=(0,)),
            TransposeLayer(1, 2),
            MaskedNormLayer,
            TransposeLayer(1, 2),
            nn.ReLU(True),
            nn.Conv1d(dModel, outD, kernel_size=(1,), stride=(1,), padding=(0,))
        )

    def forward(self, inputBatch):
        return self.conv(inputBatch)
    
class V2V(nn.Module):

    def __init__(self, dropout_features, frontend, output_size, reqInpLen, ALPHA, eosIdx, spaceIdx):
        super(V2V, self).__init__()
        self.feature_extractor = VisFeatureExtractionModel(frontend)
        self.dropout_feats = nn.Dropout(p=dropout_features)

        self.vocab_size = output_size
        self.numClasses = output_size
        
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

        # self.att_deocder = att_deocder(
        #     encode_size=out_dim,
        #     dec_dim=512,
        #     att_dim=1024,
        #     vocab_size=output_size,
        #     init_adadelta=True,
        #     ctc_weight=0.5,
        #     attention="dot",
        #     decoder="LSTM",
        #     emb_drop=0,
        # )
        self.frontend = frontend
        dModel = 512

        self.CTCLossFunction = [SmoothCTCLoss(output_size, blank=0)]
        self.CELossFunction = [SmoothCrossEntropyLoss()]
        self.reqInpLen = reqInpLen
        self.alpha = ALPHA
        self.eosIdx = eosIdx
        self.spaceIdx = spaceIdx
        self.EncoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=500) #peMaxLen
        tx_norm = nn.LayerNorm(dModel)
        videoEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.videoEncoder = nn.TransformerEncoder(videoEncoderLayer, num_layers=6, norm=tx_norm)
        self.maskedLayerNorm = MaskedLayerNorm()
        self.videoConv = conv1dLayers(self.maskedLayerNorm, out_dim, dModel, dModel)
        self.jointOutputConv = outputConv(self.maskedLayerNorm, dModel, output_size)
        self.decoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=500)
        self.embed = torch.nn.Sequential(
            nn.Embedding(output_size, dModel),
            self.decoderPositionalEncoding
        )
        jointDecoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.jointAttentionDecoder = nn.TransformerDecoder(jointDecoderLayer, num_layers=6, norm=tx_norm)
        self.jointAttentionOutputConv = outputConv("LN", dModel, output_size)

    def forward(self, source, target_lengths, targetinBatch, teacher_forcing=0):
        _, _, source, vidLen = source
        # frames = source.shape[1]
        x = self.feature_extractor(source, vidLen)  # (B*N) x D
        x = self.dropout_feats(x)
        # x = self.backbone.feature_aggregator(x)     # B x D x N
        # x = self.backbone.dropout_agg(x)
        vsr = None
        att = None

        x = list(torch.split(x, vidLen.tolist(), dim=0))
        x, inputLenBatch, mask = self.makePadding(x, vidLen)
        # x: b*T*D

        if isinstance(self.maskedLayerNorm, MaskedLayerNorm):
            self.maskedLayerNorm.SetMaskandLength(mask, inputLenBatch)

        if self.frontend == 'resnet18':
            x = self.EncoderPositionalEncoding(x.permute(1,0,2))
        elif self.frontend == "resnet50":
            x = x.transpose(1, 2)
            x = self.videoConv(x)
            x = x.transpose(1, 2).transpose(0, 1)
            x = self.EncoderPositionalEncoding(x)#.permute(1,0,2)

        x = self.videoEncoder(x, src_key_padding_mask=mask)
        vsr = self.jointOutputConv(x.permute(1,2,0))
        vsr = F.log_softmax(vsr.permute(2,0,1), dim=-1)
        targetinBatch = self.embed(targetinBatch.transpose(0, 1))
        targetinMask = self.makeMaskfromLength(targetinBatch.shape[:-1][::-1], target_lengths, x.device)
        squareMask = generate_square_subsequent_mask(targetinBatch.shape[0], x.device)
        att = self.jointAttentionDecoder(targetinBatch, x, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask, memory_key_padding_mask=mask)
        # att: T*B*D
        att = self.jointAttentionOutputConv(att.permute(1,2,0)) # input: B*D*T, output: B*V*T
        att = att.permute(0,2,1) # B*T*V

        return inputLenBatch, (vsr, att)

    def inference(self, inputBatch, Lambda, beamWidth, eosIx, blank):
        _, _, source, vidLen = inputBatch
        # frames = source.shape[1]
        x = self.feature_extractor(source, vidLen)  # (B*N) x D
        x = self.dropout_feats(x)

        device = source.device
        eosIx = self.eosIdx
        blank = self.spaceIdx

        x = list(torch.split(x, vidLen.tolist(), dim=0))
        encodedBatch, inputLenBatch, mask = self.makePadding(x, vidLen)
        # x: b*T*D

        if isinstance(self.maskedLayerNorm, MaskedLayerNorm):
            self.maskedLayerNorm.SetMaskandLength(mask, inputLenBatch)

        encodedBatch = self.EncoderPositionalEncoding(encodedBatch.permute(1,0,2))
        encodedBatch = self.videoEncoder(encodedBatch, src_key_padding_mask=mask)

        CTCOutputConv = self.jointOutputConv
        attentionDecoder = self.jointAttentionDecoder
        attentionOutputConv = self.jointAttentionOutputConv

        CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2)
        CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
        CTCOutputBatch = CTCOutputBatch.transpose(1, 2)
        # claim batch and time step
        batch = CTCOutputBatch.shape[0]
        T = inputLenBatch.cpu()
        # claim CTClogprobs and Length
        CTCOutputBatch = CTCOutputBatch.cpu()
        CTCOutLogProbs = F.log_softmax(CTCOutputBatch, dim=-1)
        predictionLenBatch = torch.ones(batch, device=device).long()
        # init Omega and Omegahat for attention beam search
        Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in range(batch)]
        Omegahat = [[] for i in range(batch)]
        # init
        gamma_n = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()
        gamma_b = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()
        for b in range(batch):
            gamma_b[b, 0, 0, 0] = 0
            for t in range(1, T[b]):
                gamma_n[b, t, 0, 0] = -np.inf
                gamma_b[b, t, 0, 0] = 0
                for tao in range(1, t + 1):
                    gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[b, tao, blank]

        newhypo = torch.arange(1, self.numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

        for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
            predictionBatch = []
            for i in range(batch):
                predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]
                Omega[i].append([])
            predictionBatch = torch.stack(predictionBatch).long().to(device)
            predictionBatch = self.embed(predictionBatch.transpose(0, 1))
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool()
            if not predictionBatch.shape[1] == encodedBatch.shape[1]:
                encoderIndex = [i for i in range(batch) for j in range(beamWidth)]
                encodedBatch = encodedBatch[:, encoderIndex, :]
                mask = mask[encoderIndex]
                predictionLenBatch = predictionLenBatch[encoderIndex]
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1)
            attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu()

            # Decode
            h = []
            alpha = []
            for b in range(batch):
                h.append([])
                alpha.append([])
                for o in Omega[b][l - 1][:beamWidth]:
                    h[b].append([o[0].tolist()])
                    alpha[b].append([[o[1], o[2]]])
            h = torch.tensor(h)
            alpha = torch.tensor(alpha).float()
            numBeam = alpha.shape[1]
            recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1)
            h = torch.cat((torch.repeat_interleave(h, self.numClasses - 1, dim=2), recurrnewhypo), dim=-1)
            alpha = torch.repeat_interleave(alpha, self.numClasses - 1, dim=2)
            alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1)

            # h = (batch * beam * 39 * hypoLength)
            # alpha = (batch * beam * 39)
            # CTCOutLogProbs = (batch * sequence length * 40)
            # gamma_n or gamma_b = (batch * max time length * beamwidth * 40 <which is max num of candidates in one time step>)
            CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, self.numClasses - 1, blank, eosIx)
            alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]
            hPaddingShape = list(h.shape)
            hPaddingShape[-2] = 1
            h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)

            activeBatch = (l < T).nonzero().squeeze(-1).tolist()
            for b in activeBatch:
                for i in range(numBeam):
                    Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))

            alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)
            alpha[:, :, -1, 0] = -np.inf
            predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices
            for b in range(batch):
                for pos, c in enumerate(predictionRes[b]):
                    beam = c // self.numClasses
                    c = c % self.numClasses
                    Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                    gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                    gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
            gamma_n[:, :, :, 1:] = -np.inf
            gamma_b[:, :, :, 1:] = -np.inf
            predictionLenBatch += 1

        predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]
        predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
        return torch.cat([prediction[1:] for prediction in predictionBatch]).int(), torch.tensor(predictionLenBatch).int()

    def attentionAutoregression(self, inputBatch, maskw2v, device, eosIx):
        encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v)
        attentionDecoder = self.jointAttentionDecoder
        attentionOutputConv = self.jointAttentionOutputConv

        # claim batch and time step
        batch = encodedBatch.shape[1]
        T = inputLenBatch.cpu()
        # claim CTClogprobs and Length
        predictionLenBatch = torch.ones(batch, device=device).long()
        endMask = torch.ones(batch, device=device).bool()
        predictionInpBatch = torch.full((batch, 1), eosIx, device=device).long()

        while endMask.max() and predictionLenBatch.max() < T.max():
            predictionBatch = self.embed(predictionInpBatch.transpose(0, 1))
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool()
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1)
            predictionNewBatch = torch.argmax(attentionOutputBatch, dim=-1) + 1
            endMask *= ~(predictionNewBatch == eosIx)
            predictionNewBatch = predictionNewBatch.unsqueeze(0).transpose(0, 1)
            predictionInpBatch = torch.cat((predictionInpBatch, predictionNewBatch), dim=-1)
            predictionLenBatch[endMask] += 1
        predictionInpBatch = torch.cat((predictionInpBatch, torch.full((batch, 1), eosIx, device=device)), dim=-1)
        return torch.cat([predictionInp[1:predictionLenBatch[b] + 1] for b, predictionInp in enumerate(predictionInpBatch)]).int().cpu(), \
               predictionLenBatch.int().cpu()

    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask

    def makePadding(self, videoBatch, vidLen):
        vidPadding = torch.zeros(len(videoBatch)).long().to(vidLen.device)

        mask = (vidPadding + vidLen) > self.reqInpLen
        vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)

        vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
        vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()

        for i, _ in enumerate(videoBatch):
            pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
            videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        videoBatch = pad_sequence(videoBatch, batch_first=True)
        inputLenBatch = (vidLen + vidPadding).long()
        mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, videoBatch.device)

        return videoBatch, inputLenBatch, mask