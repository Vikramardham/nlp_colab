#models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot
torch.manual_seed(1)
import wandb
# Custom Dataset
class mydataset(Dataset):
    
    def __init__(self, x, y):
        super(mydataset, self).__init__()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        #self.x = transforms.ToTensor()(x)
        #self.y = transforms.ToTensor()(y)
    
    def __len__(self):
        return self.x.size()[0]
        
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    
# PyTorch model
class SimpleRNN(nn.Module):
    
    def __init__(self, emb_enc, seq_ln, emb_sz, hd_sz, vocab_sz):
        super(SimpleRNN, self).__init__()
        
        self.output_sz = 5
        #self.inp_sz = inp_sz
        self.emb_sz = emb_sz
        self.hd_sz = hd_sz
        
        #self.emb = nn.Embedding(vocab_sz, emb_sz)
        self.emb = emb_enc
        #self.dropout = nn.Dropout(p=dp)
        self.i2h = nn.Linear(emb_sz, self.hd_sz)
        self.h2h = nn.Linear(hd_sz, hd_sz)
        self.h = torch.zeros(1, self.hd_sz)
        self.h2o = nn.Linear(hd_sz*seq_ln, 5)
        self.bn = nn.BatchNorm1d(5)
        self.dp = nn.Dropout(p=0.2)
    def forward(self, x):
        emb = self.emb(x)
        h = self.h
        out = []

        for xi in torch.transpose(emb, 0, 1):
            i2h = self.i2h(xi)
            h2h = torch.tanh(i2h + self.h2h(h))
            out.append(h2h)
        y = torch.flatten(torch.stack(out, dim=1), start_dim=1)
        y = F.log_softmax(self.bn(self.h2o(y)), dim=1)
        return y
#GRU 
def rnn_loop(cell, h, x):
    res = []
    for x_ in x.transpose(0,1):
        #print('x_', x_.size())
        h = cell(x_, h)
        res.append(h)
    return torch.stack(res, dim=1)

class GRUCell(nn.Module):
    def __init__(self, ni, nh):
        super(GRUCell, self).__init__()
        self.ni, self.nh = ni, nh
        self.i2h = nn.Linear(ni, 3*nh)
        self.h2h = nn.Linear(nh, 3*nh)
    
    def forward(self, x, h):
        gate_x = self.i2h(x).squeeze()
        gate_h = self.h2h(h).squeeze()
        i_r,i_u,i_n = gate_x.chunk(3, 1)
        h_r,h_u,h_n = gate_h.chunk(3, 1)
        
        #print(gate_x.size(), gate_h.size(), i_r.size(), h_r.size(), x.size())
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_u + h_u)
        newgate = torch.tanh(i_n + (resetgate*h_n))
        return updategate*h + (1-updategate)*newgate
    
class SimpleGRU(nn.Module):
    def __init__(self, emb_enc, seq_ln, emb_sz, hd_sz, vocab_sz, bs= 64):
        super(SimpleGRU, self).__init__()
        #self.i_h = nn.Embedding(vocab_sz, emb_sz)
        self.i_h = emb_enc
        self.h_o = nn.Linear(hd_sz*seq_ln, 5)
        self.bn = nn.BatchNorm1d(5)
        self.hd_sz = hd_sz
        self.rnnc = GRUCell(emb_sz, hd_sz)
        
    def forward(self, x):
        #print(emb_sz)
        #print('input:', self.i_h(x).size())
        #self.h = torch.zeros(1, )
        i_h = self.i_h(x)
        h = torch.zeros(1, i_h.size()[0], self.hd_sz)
        res = rnn_loop(self.rnnc, h, i_h)
        self.h = res[:, -1].detach()
        res = torch.transpose(torch.squeeze(res), 0, 1)
        out = self.bn(self.h_o(torch.flatten(res, start_dim=1)))
        return F.log_softmax(out, dim=1)

class TorchRNN(nn.Module):
    def __init__(self, emb_sz, inp_sz, vocab_sz, hd_sz, out_sz):
        super(TorchRNN, self).__init__()
        self.inp_sz = inp_sz
        self.hd_sz = hd_sz
        
        self.rnn = nn.Sequential(nn.Embedding(vocab_sz, emb_sz),
                          nn.RNN(input_size=emb_sz, hidden_size=hd_sz, batch_first=True))
        self.out = nn.Linear(inp_sz* hd_sz, out_sz)
        self.activ = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        rnn_out = self.rnn(x)
        #print(rnn_out[0].reshape(-1, inp_sz*hd_sz).size())
        final_out = self.activ(self.out(rnn_out[0].reshape(-1, self.inp_sz*self.hd_sz)))
        #print(final_out.size())
        return final_out

class TorchGRU(nn.Module):
    def __init__(self, emb_sz, inp_sz, vocab_sz, hd_sz, out_sz):
        super(TorchRNN, self).__init__()
        self.inp_sz=inp_sz
        self.hd_sz = hd_sz
        self.rnn = nn.Sequential(nn.Embedding(vocab_sz, emb_sz),
                          nn.RNN(input_size=emb_sz, hidden_size=hd_sz, batch_first=True))
        self.out = nn.Linear(inp_sz* hd_sz, out_sz)
        self.activ = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        rnn_out = self.rnn(x)
        #print(rnn_out[0].reshape(-1, inp_sz*hd_sz).size())
        final_out = self.activ(self.out(rnn_out[0].reshape(-1, self.inp_sz*self.hd_sz)))
        #print(final_out.size())
        return final_out
#class SimpleGRU(nn.Module):
#    
#    def __init__(self, inp_sz, emb_sz, hd_sz, output_sz, vocab_sz):
#        super(SimpleGRU, self).__init__()
#        
#        self.output_sz = output_sz
#        self.inp_sz = inp_sz
#        self.emb_sz = emb_sz
#        self.hd_sz = hd_sz
        
#        self.emb = nn.Embedding(vocab_sz, emb_sz)
#        self.dropout = nn.Dropout(p=dp)
        
#        self.update = nn.Linear(self.emb_sz+self.hd_sz, self.hd_sz)
#        self.reset = nn.Linear(self.emb_sz+self.hd_sz, self.hd_sz)
#        self.output = nn.Linear(self.emb_sz+self.hd_sz, self.hd_sz)
#        self.final = nn.Linear(self.inp_sz*self.hd_sz, self.output_sz)
#        self.h = torch.zeros(1, hd_sz)
#        self.activ = nn.LogSoftmax(dim=1)
                             
#    def forward(self, x):
        
#        emb = self.dropout(self.emb(x))
#        h = self.h.view(1, -1).repeat(x.size()[0], 1)
#        out=[]
        
#        for xi in torch.transpose(emb, 0, 1):
#            combined = torch.cat( (xi, h), dim=1)
#            zt = torch.sigmoid(self.update(combined))
#            reset = torch.sigmoid(self.reset(combined))
#            output = torch.tanh(self.output(combined))
            
#            h = zt*h + (1-zt)*output
#            out.append(output)
            
#        y = torch.cat(out).view(-1, self.inp_sz*self.hd_sz)
        
        #print(y.size())
#        return self.activ(self.final(y))
