import io
import pickle
import numpy as np
import pandas as pd
import random
from tqdm.notebook import tqdm 
import wandb

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from nlp_models import *
from nlp_train import *
from torchsummary import summary

from fastai import *
from fastai.text import *
from fastai.tabular import *
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
import spacy
import fasttext as ft

Nwords = 25
seq_ln = Nwords
emb_sz = 300
hd_sz = 10
output_sz = 5

# read_data
emb_enc = torch.load('en_emb.pth')
wiki_words = pickle.load(open('../../data/external/itos_wt103.pkl','rb'))

def pipeline(xin, yin):
    return mydataset([pad_zeros(wiki_vocab.numericalize(i)) for i in tokenizer.process_all(xin)], yin)

def pad_zeros(inp, max_len=Nwords):
    ''' pad zeros if the len(input) < max_len'''
    if len(inp)>=max_len:
        return inp[:max_len]
    else:
        return inp+[0]*(max_len-len(inp))
def create_emb(vecs, itos, em_sz=10, mult=1.):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    vec_dic = {w:vecs.get_word_vector(w) for w in vecs.get_words()}
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = tensor(vec_dic[w])
        except: miss.append(w)
    return emb


categories = ['sci.crypt', 'sci.electronics',
              'sci.med', 'sci.space', 'soc.religion.christian']
newsgroups_all= fetch_20newsgroups(subset='all',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
X_train, X_test, y_train, y_test = train_test_split(newsgroups_all.data, newsgroups_all.target,
                                         test_size=0.2, stratify=newsgroups_all.target)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                         test_size=0.5, stratify=y_test)
tokenizer = Tokenizer()
spc = SpacyTokenizer('en')

wiki_vocab = Vocab.create([wiki_words], max_vocab=60000, min_freq=1)
token_train = tokenizer.process_all(X_train)
vocab_sz = len(wiki_vocab.itos)

xtrain = [pad_zeros(wiki_vocab.numericalize(i)) for i in token_train]
word_to_ix = {word: i for i, word in enumerate(wiki_vocab.itos)}

valid =pipeline(X_valid, y_valid)
test = pipeline(X_test, y_test)
train_data = mydataset(xtrain, y_train)

del X_valid
del X_test
del X_train
del y_train
del xtrain



    