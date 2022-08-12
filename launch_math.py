# Imports
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.utils import set_seed, give_exam
from mingpt.data_wrappers import AdditionDataset
from mingpt.model import GPT, GPTConfig, GPT1Config
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


# make deterministic
set_seed(42)


# create a dataset for e.g. 2-digit addition
ndigit = 2
train_dataset = AdditionDataset(ndigit=ndigit, split='train')
test_dataset = AdditionDataset(ndigit=ndigit, split='test')


# initialize a baby GPT model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
                  n_layer=2, n_head=4, n_embd=128)

model = GPT(mconf)

input("Model created: continue to training?")


# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=50, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset)*(ndigit+1),
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()


input("Training complete: continue to testing?")

# now let's give the trained model an addition exam
# training set: how well did we memorize?
give_exam(ndigit, model, trainer, train_dataset, batch_size=1024, max_batches=10)

# test set: how well did we generalize?
give_exam(ndigit, model, trainer, test_dataset, batch_size=1024, max_batches=-1)
