import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class outputConv(nn.Module):
    def __init__(self, MaskedNormLayer, dModel, numClasses):
        super(outputConv, self).__init__()
        if MaskedNormLayer == "LN":
            self.outputconv = nn.Sequential(
                nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel // 2),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel // 2),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, numClasses, kernel_size=(1,), stride=(1,), padding=(0,))
            )
        else:
            self.outputconv = nn.Sequential(
                nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, numClasses, kernel_size=(1,), stride=(1,), padding=(0,))
            )

    def forward(self, inputBatch):
        return self.outputconv(inputBatch)
    

class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, inputBatch):
        return inputBatch.transpose(self.dim1, self.dim2)
    

class MaskedLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(MaskedLayerNorm, self).__init__()
        # self.register_buffer('mask', None, persistent=False)
        # self.register_buffer('inputLenBatch', None, persistent=False)
        self.eps = eps

    def SetMaskandLength(self, mask, inputLenBatch):
        self.mask = mask
        self.inputLenBatch = inputLenBatch

    def expand2shape(self, inputBatch, expandedShape):
        return inputBatch.unsqueeze(-1).unsqueeze(-1).expand(expandedShape)

    def forward(self, inputBatch):
        dModel = inputBatch.shape[-1]
        maskBatch = ~self.mask.unsqueeze(-1).expand(inputBatch.shape)

        meanBatch = (inputBatch * maskBatch).sum((1, 2)) / (self.inputLenBatch * dModel)
        stdBatch = ((inputBatch - self.expand2shape(meanBatch, inputBatch.shape)) ** 2 * maskBatch).sum((1, 2))
        stdBatch = stdBatch / (self.inputLenBatch * dModel)

        # Norm the input
        normed = (inputBatch - self.expand2shape(meanBatch, inputBatch.shape)) / \
                 (torch.sqrt(self.expand2shape(stdBatch + self.eps, inputBatch.shape)))
        return normed
    

def generate_square_subsequent_mask(sz: int, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float() * (math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position / denominator)
        pe[:, 1::2] = torch.cos(position / denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0], :, :]
        return outputBatch
    

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]