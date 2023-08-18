import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import numpy as np


class att_deocder(nn.Module):
    def __init__(self, encode_size, dec_dim, att_dim, vocab_size, init_adadelta=True, ctc_weight=0.5, attention="dot", decoder="LSTM", emb_drop=0.0):
        super(att_deocder, self).__init__()

        # Setup
        assert 0 <= ctc_weight <= 1
        self.input_size = encode_size
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1

        # Modules
        if self.enable_ctc:
            self.ctc_layer = nn.Linear(self.input_size, vocab_size)
        if self.enable_att:
            self.dec_dim = dec_dim
            # self.pre_embed = nn.Embedding(vocab_size, vocab_size)
            self.embed_drop = nn.Dropout(emb_drop)
            self.decoder = Decoder(
                self.input_size+dec_dim, decoder, dec_dim, layer=2, dropout=0.2)
            query_dim = self.dec_dim*self.decoder.layer
            self.attention = Attention(
                self.input_size, query_dim, attention, att_dim, 6, 0.5, False, 100, 10)
        self.tgt_embedding = nn.Linear(vocab_size,self.dec_dim)
        self.att_proj = nn.Linear(self.dec_dim,vocab_size)


        # Init
        if init_adadelta:
            self.apply(init_weights)
            if self.enable_att:
                for l in range(self.decoder.layer):
                    bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
                    bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        self.decoder.set_state(prev_state)
        self.attention.set_mem(prev_attn)

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
            self.encoder.sample_rate))
        if self.encoder.vgg:
            msg.append(
                '           | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.encoder.cnn:
            msg.append(
                '           | CNN Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.enable_ctc:
            msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
                self.ctc_weight))
        if self.enable_att:
            msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(
                self.attention.mode, 1-self.ctc_weight))
        return msg

    def forward(self, context_feature, feature_len, decode_step, tf_rate=0.0, teacher=None,
                emb_decoder=None, get_dec_state=False):
        '''
        Arguments
            context_feature - [BxTxD] Context vector with shape 
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps 
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxL]   Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
        bs = context_feature.shape[0]
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None

        if teacher is not None and tf_rate>0:
            teacher = self.tgt_embedding(teacher.float())

        # CTC based decoding
        if self.enable_ctc:
            # ctc_output = F.log_softmax(self.ctc_layer(context_feature), dim=-1)
            ctc_output = self.ctc_layer(context_feature)

        # Attention based decoding
        if self.enable_att:
            # Init (init char = <SOS>, reset all rnn state and cell) (<s>)
            self.decoder.init_state(bs)
            self.attention.reset_mem()
            if teacher is not None and tf_rate>0:
                last_char = teacher[:,0,:]
            else:
                last_char = torch.zeros((bs,40),dtype=context_feature.dtype,device=context_feature.device)
                last_char[:,0] = 1
                last_char = self.tgt_embedding(last_char)
            att_seq, output_seq = [], []

            # Preprocess data for teacher forcing
            # if teacher is not None:
            #     teacher = self.embed_drop(self.pre_embed(teacher))

            # Decode
            for t in range(decode_step):
                # Attend (inputs current state of first layer, encoded features)
                attn, context = self.attention(
                    self.decoder.get_query(), context_feature, feature_len)
                # Decode (inputs context + embedded last character)
                decoder_input = torch.cat([last_char, context], dim=-1)
                cur_char, d_state = self.decoder(decoder_input)
                # Prepare output as input of next step
                if (teacher is not None and tf_rate>0 and t<decode_step-1):
                    # Training stage
                    if (tf_rate == 1) or (torch.rand(1).item() <= tf_rate):
                        # teacher forcing
                        last_char = teacher[:, t+1, :]
                    else:
                        # self-sampling (replace by argmax may be another choice)
                        with torch.no_grad():
                            if (emb_decoder is not None) and emb_decoder.apply_fuse:
                                _, cur_prob = emb_decoder(
                                    d_state, cur_char, return_loss=False)
                            else:
                                cur_prob = cur_char.softmax(dim=-1)
                            sampled_char = Categorical(cur_prob).sample()
                        last_char = self.embed_drop(sampled_char)
                else:
                    # Inference stage
                    if (emb_decoder is not None) and emb_decoder.apply_fuse:
                        _, cur_char = emb_decoder(
                            d_state, cur_char, return_loss=False)
                    # argmax for inference
                    last_char = cur_char

                # save output of each step
                output_seq.append(cur_char)
                att_seq.append(attn)
                if get_dec_state:
                    dec_state.append(d_state)
                
                # # <eos> stop
                # if torch.argmax(self.att_proj(cur_char),dim=-1)==2:
                #     break

            att_output = self.att_proj(torch.stack(output_seq, dim=1))  # BxTxV
            att_seq = torch.stack(att_seq, dim=2)       # BxNxDtxT
            if get_dec_state:
                dec_state = torch.stack(dec_state, dim=1)

        return ctc_output, att_output, att_seq, dec_state
    

def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3, 4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError
            
        
def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias


class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''
    # ToDo:　More elegant way to implement decoder

    def __init__(self, input_dim, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout

        # Init
        assert module in ['LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'

        # Modules
        self.layers = getattr(nn, module)(
            input_dim, dim, num_layers=layer, dropout=dropout, batch_first=True)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer, bs, self.dim), device=device),
                                 torch.zeros((self.layer, bs, self.dim), device=device))
        else:
            self.hidden_state = torch.zeros(
                (self.layer, bs, self.dim), device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(
                device), hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(), self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.dim*self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim*self.layer)

    def forward(self, x):
        ''' Decode and transform into vocab '''
        if not self.training:
            self.layers.flatten_parameters()
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)
        # char = self.char_trans(self.final_dropout(x))
        # return char, x
        return x, self.hidden_state


class Attention(nn.Module):
    ''' Attention mechanism
        please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
        Input : Decoder state                      with shape [batch size, decoder hidden dimension]
                Compressed feature from Encoder    with shape [batch size, T, encoder feature dimension]
        Output: Attention score                    with shape [batch size, num head, T (attention score of each time step)]
                Context vector                     with shape [batch size, encoder feature dimension]
                (i.e. weighted (by attention score) sum of all timesteps T's feature) '''

    def __init__(self, v_dim, q_dim, mode, dim, num_head, temperature, v_proj,
                 loc_kernel_size, loc_kernel_num):
        super(Attention, self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear(q_dim, dim*num_head)
        self.proj_k = nn.Linear(v_dim, dim*num_head)
        self.v_proj = v_proj
        if v_proj:
            self.proj_v = nn.Linear(v_dim, v_dim*num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(
                loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim*num_head, v_dim)

        # Stored feature
        self.key = None
        self.value = None
        self.mask = None

    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def set_mem(self, prev_attn):
        self.att_layer.set_mem(prev_attn)

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs, ts, _ = enc_feat.shape
        query = torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(
            bs*self.num_head, self.dim)  # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_feat, enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key = torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(
                enc_feat)) if self.v_proj else enc_feat  # BxTxN

            if self.num_head > 1:
                self.key = self.key.view(bs, ts, self.num_head, self.dim).permute(
                    0, 2, 1, 3)  # BxNxTxD
                self.key = self.key.contiguous().view(bs*self.num_head, ts, self.dim)  # BNxTxD
                if self.v_proj:
                    self.value = self.value.view(
                        bs, ts, self.num_head, self.v_dim).permute(0, 2, 1, 3)  # BxNxTxD
                    self.value = self.value.contiguous().view(
                        bs*self.num_head, ts, self.v_dim)  # BNxTxD
                else:
                    self.value = self.value.repeat(self.num_head, 1, 1)

        # Calculate attention
        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head > 1:
            context = context.view(
                bs, self.num_head*self.v_dim)    # BNxD  -> BxND
            context = self.merge_head(context)  # BxD

        return attn, context
    

class BaseAttention(nn.Module):
    ''' Base module for attentions '''

    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self, prev_att):
        pass

    def compute_mask(self, k, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs, ts, _ = k.shape
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn)  # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(
            1)  # BNxT x BNxTxD-> BNxD
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(
            1, 2)).squeeze(1)  # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy, v)

        attn = attn.view(-1, self.num_head, ts)  # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''

    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att = None
        self.loc_conv = nn.Conv1d(
            num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim, bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh, ts, _ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs, self.num_head, ts)).to(k.device)
            for idx, sl in enumerate(self.k_len):
                self.prev_att[idx, :, :sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(
            self.prev_att).transpose(1, 2)))  # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(
            1, self.num_head, 1, 1).view(-1, ts, self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1)  # BNx1xD

        # Compute energy and context
        energy = self.gen_energy(torch.tanh(
            k+q+loc_context)).squeeze(2)  # BNxTxD -> BNxT
        output, attn = self._attend(energy, v)
        attn = attn.view(bs, self.num_head, ts)  # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn