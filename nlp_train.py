import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import accuracy_score as accuracy, f1_score, \
precision_score, roc_auc_score as roc_score, recall_score, \
balanced_accuracy_score as bal_accuracy

class SimpleTrain():
  '''
  A simple training protocol for training, logging and saving models
  '''
  def __init__(self, model, loss_func, folder='gru_classification', title='gru'):
    super().__init__()
    self.epochs = 10
    self.func = loss_func
    self.model = model
    self.optim = optim.Adam(model.parameters(), lr=1e-02)
    self.bs = 32
    self.lr = 2e-03
    self.metric_dict={'acc':acc_torch, 'bal_acc': bal_accuracy, 'precision':precision,
            'f1': f1, 'roc': roc, 'recall': recall}
    self.folder =folder
    self.title = title
      
  def train(self, train_data, valid, config,
            metrics=['acc', 'precision', 'recall'], log=True, sch=None):
    dataloader = DataLoader(train_data, batch_size=config['bs'], shuffle=True, drop_last=True)
    
    if config:
      #self.optim = optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
      #self.optim = optim.SGD(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'], 
      #momentum=0.9, nesterov=True)
      self.optim = optim.Adagrad(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
      self.bs=config['bs']
      self.epochs=config['epochs']
    
    if sch=='cos':
      scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epochs)
    elif sch=='cyclic':
      scheduler = optim.lr_scheduler.CyclicLR(self.optim, max_lr=1e-01, base_lr=1e-4,)
    else:
      scheduler = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=2e-01, steps_per_epoch=len(dataloader), epochs=self.epochs)
    best = -1
    
    print('----------------------------------------------------')
    p1= 'Epoch:\t Train_loss \t Valid_loss\t'
    for metric in metrics:
      p1 = p1 + f'{metric}'+ '\t'
    print(p1)

    #writer=SummaryWriter(f'experiment')
    writer=SummaryWriter(comment=f'**config')

    for epoch in range(self.epochs):
      self.model.train()
      
      for x, y_ in dataloader :
        y = self.model(x)
        loss = self.func(y, y_)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
      scheduler.step()
          
      self.model.eval()
      with torch.no_grad():
        yout = self.model(valid.x)
        validloss = self.func(yout, valid.y)
        validout=[]
        trainout=[]
        
        for metric in metrics:
          #print(self.metric_dict[metric])
          validout.append(self.metric_dict[metric](valid.y, torch.argmax(yout, dim=1)))
          trainout.append(self.metric_dict[metric](train_data.y, torch.argmax(self.model(train_data.x), dim=1)))
      
          is_best = validout[0]>best
          best = max(best, validout[0])
            
        if is_best:
          state = dict({'epoch': epoch + 1, 'arch': self.model,'model_state':self.model.state_dict(),
          'state_dict': self.model.state_dict(),
          'best_acc1': best, 'optimizer' : self.optim.state_dict()})
          #torch.save(state, f'../models/{self.folder}/{self.title}_model.pth')
      
        p2 = f'{epoch}\t {loss.item():3.4f} \t {validloss.item():3.4f} \t '
        
        for oo in validout:
          #print(oo)
          p2 = p2 + f'{oo:3.4f}'+ '\t'

        for oo in trainout:
          p2 = p2 + f'{oo:3.4f}'+'\t'
        print(p2)
        
        print('---------------------------------------------')
        
        if log:
          writer.add_scalars('run', {'loss':loss.item(), 'valid_loss':validloss.item()}, epoch)
          #logging={'epoch': epoch, 'loss': loss, 'valid_loss': loss, **dict(zip(metrics, validout))}
          #wandb.log(logging)

    return 

def to_numpy(func):
  def wrapper(*args):
      args=[arg.data.numpy() for arg in args]
      return func(*args)
  return wrapper
@to_numpy
def acc_torch(targets, output):
  return accuracy(targets, output)
@to_numpy
def bal_acc(targets, output):
  return bal_accuracy(targets, output)
@to_numpy
def f1(targets, output):
  return f1_score(targets, output)
@to_numpy
def precision(targets, output):
  return precision_score(targets, output, average='weighted')
@to_numpy
def roc(targets, output):
  return roc_score(targets, output)
@to_numpy
def recall(targets, output):
  return recall_score(targets, output, average='weighted')
