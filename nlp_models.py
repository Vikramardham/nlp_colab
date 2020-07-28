'''
File Defining RNN Architectures and Custom Datasets for RNN and Transformer Models
Vikram Reddy Ardham
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class mydataset(Dataset):
  '''
  Pytorch Dataset class to create a a dataloader
  '''
  def __init__(self, x, y):
    super().__init__()
    self.x = torch.tensor(x, dtype=torch.long)
    self.y = torch.tensor(y)
  
  def __len__(self):
    return self.x.size()[0]
      
  def __getitem__(self, ix):
    return self.x[ix], self.y[ix]

class CustomDataset(Dataset):
  '''
  Custom Dataset that tokenizes input text data for the transformer model
  '''
  def __init__(self, x, y, tokenizer, max_len):
    self.tokenizer = tokenizer
    self.data = x
    self.targets = y
    self.maxlen = max_len
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    inputs = self.tokenizer.encode_plus(self.data[idx], add_special_tokens=True, 
                                        max_length=self.maxlen, pad_to_max_length=True,
                                        return_token_type_ids=True, truncation=True)
    return {'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask':torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids':torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets':torch.tensor(self.targets[idx], dtype=torch.long)}

class SpatialDropout(nn.Dropout2d):
  '''
  Spatial Dropout for the embedding layer
  Refactor the existing Dropou2d class in PyTorch
  '''
  def forward(self, inps):
    inps =inps.unsqueeze(2)
    inps = inps.permute(0, 3, 2, 1)
    inps = super().forward(inps)
    inps = inps.permute(0, 3, 2, 1)
    inps = inps.squeeze(2)
    return inps

class myArch(nn.Module):
  '''Basic RNN based Text-classification architecture'''
  def __init__(self, emb_enc, seq_ln, hd_sz, stacks=2, CHOICE='LSTM'):
    super().__init__()
    self.emb = emb_enc
    self.seq_ln = seq_ln
    self.hd_sz = hd_sz
    self.emb_sz = emb_enc.embedding_dim
    self.bidirectional = False
    self.emb_dropout = SpatialDropout(p=0.4)
    
    # Choice of the type of RNN
    ARCH = {'LSTM': nn.LSTM(input_size=self.emb_sz, 
                          hidden_size=self.hd_sz,
                          num_layers =stacks, bias=True, 
                          bidirectional=self.bidirectional, dropout=0.8,
                          batch_first=True), 
              'GRU': SimpleGRU(self.seq_ln, self.emb_sz, self.hd_sz) , 
              'Vanilla RNN': SimpleRNN(self.seq_ln, self.emb_sz, self.hd_sz), }
      
    self.RNN = ARCH[CHOICE]
    
    self.dropout = nn.Dropout(p=0.4)
    Nout = 128
    self.linear = nn.Linear(self.hd_sz*2*(1+int(self.bidirectional)), Nout)
    self.batchnorm = nn.BatchNorm1d(Nout)
    self.out = nn.Linear(Nout, 5)

  def forward(self, X):
    '''
    1. Embedding layer
    2. Spatial Dropout
    3. RNN layer
    4. concatenate (avg_pool, max_pool)
    5. Linear Layer
    5. Batch Normalization
    6. Dropout
    7. Linear Layer to output class probabilities (5 classes here)
    '''
    h = self.emb(X)
    h = self.emb_dropout(h)
    gru_out, _ = self.RNN(h)

    avg_pool = torch.mean(gru_out, 1)
    max_pool, _ = torch.max(gru_out, 1)
    conc = torch.cat((avg_pool, max_pool), 1)
    
    conc = self.batchnorm(nn.ReLU()(self.linear(conc)))
    conc = self.dropout(conc)
    out = self.out(conc)
    
    return out

# My RNN
class SimpleRNN(nn.Module):
  '''Vanilla RNN implemented from Scracth '''
  def __init__(self, seq_ln, emb_sz, hd_sz):
    super(SimpleRNN, self).__init__()
    
    self.output_sz = 5
    self.hd_sz = hd_sz
    
    self.h = torch.zeros(1, self.hd_sz)
    self.i2h = nn.Linear(emb_sz, self.hd_sz)
    self.h2h = nn.Linear(hd_sz, hd_sz)
  
  def forward(self, x):
    '''
    1. Input (embedding) to hidden
    2. Hidden to hidden
    3. Add them and apply activation
    '''
    h = self.h
    out = []
    for xi in torch.transpose(x, 0, 1):
        i2h = self.i2h(xi)
        h2h = torch.tanh(i2h + self.h2h(h))
        out.append(h2h)
    return torch.stack(out, dim=1), 1 # Dummpy output as tuple to have a consistent structure with PyTorch

#rnn_loop
def rnn_loop(cell, h, x):
  res = []
  for x_ in x.transpose(0,1):
      h = cell(x_, h)
      res.append(h)
  return torch.stack(res, dim=1)


class GRUCell(nn.Module):
  '''Inspired from the fastai-nlp course'''
  def __init__(self, ni, nh):
    super(GRUCell, self).__init__()
    self.ni, self.nh = ni, nh
    self.i2h = nn.Linear(ni, 3*nh)
    self.h2h = nn.Linear(nh, 3*nh)
  
  def forward(self, x, h):
    '''
    return a weighted mean of old and newgates
    weight is the udpategate
    '''
    gate_x = self.i2h(x).squeeze()
    gate_h = self.h2h(h).squeeze()
    i_r,i_u,i_n = gate_x.chunk(3, 1)
    h_r,h_u,h_n = gate_h.chunk(3, 1)
  
    resetgate = torch.sigmoid(i_r + h_r)
    updategate = torch.sigmoid(i_u + h_u)
    newgate = torch.tanh(i_n + (resetgate*h_n))
    return updategate*h + (1-updategate)*newgate

#My GRU   
class SimpleGRU(nn.Module):
  '''
  Combine the GRUcell with the RNN loop
  '''
  def __init__(self, seq_ln, emb_sz, hd_sz):
    super().__init__()
    self.hd_sz = hd_sz
    self.rnnc = GRUCell(emb_sz, hd_sz)
      
  def forward(self, x):
    h = torch.zeros(1, x.size()[0], self.hd_sz)
    res = rnn_loop(self.rnnc, h, x)
    self.h = res[:, -1].detach()
    res = torch.transpose(torch.squeeze(res), 0, 1)
    return res, 1 # Dummy output 1 

class BertModel(nn.Module):
  '''Bert Model Class for text-classification 
  Adding a dropout layer and linear layer to spit out 5 class probabilities'''
  def __init__(self):
    super().__init__()
    self.model_name = 'bert-base-cased'
    self.l1 = transformers.BertModel.from_pretrained(self.model_name)
    self.l2 = nn.Dropout(0.3)
    self.l3 = nn.Linear(768, 5)

  def forward(self, ids, mask, token_type_ids):
    output_1 = self.l1(ids, attention_mask = mask, token_type_ids=token_type_ids)
    output_2 = self.l2(output_1[1])
    output = self.l3(output_2)
    return output
    