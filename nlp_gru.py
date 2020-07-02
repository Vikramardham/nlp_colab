# cell'# %%'
# markdown cell '# %% [markdown]'
# %%
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from helper import *

# %%
bs = 8
rnn_model = SimpleRNN(emb_enc, seq_ln, emb_sz, hd_sz, vocab_sz)
gru_model = SimpleGRU(emb_enc, seq_ln, emb_sz, hd_sz, vocab_sz, bs=bs)
rnn_train = SimpleTrain(rnn_model, nn.CrossEntropyLoss())
gru_train = SimpleTrain(gru_model, nn.CrossEntropyLoss())

# %%
hp_defaults = dict(bs = 64, lr = 0.1, epochs = 8, wd = 1e-04)
wandb.init(project= 'nlp_basics', config=hp_defaults)
config=wandb.config
wandb.watch(gru_model)

# %%
params=dict({'lr' : 0.1, 'epochs':4, 'bs': 64, 'wd':1e-5})
config.update(params, allow_val_change=True)

gru_train.train(test, valid, config.lr, config.epochs, bs=config.bs, wd=config.wd, log=True)
