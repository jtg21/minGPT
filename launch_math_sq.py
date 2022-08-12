# Imports
import logging
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from mingpt.utils import give_dataset_exam_sq, set_seed, give_random_exam_sq
from mingpt.data_wrappers import AdditionDataset, SquareRootDataset
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
nextra = 3

train_dataset = SquareRootDataset(ndigit, nextra, split='train')
test_dataset = SquareRootDataset(ndigit, nextra, split='test')


# initialize a baby GPT model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
                  n_layer=2, n_head=4, n_embd=512)

model = GPT(mconf)

input("Model created: continue to training?")


# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=10, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset)*(ndigit+1),
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()



# input("Training complete: continue to testing?")
# print("Test on training set")
# give_exam_sq_from_dataset(ndigit, nextra, model, trainer, train_dataset)

# print("Test on test set")
# give_dataset_exam_sq(ndigit, nextra, model, trainer, test_dataset)

give_random_exam_sq(ndigit, nextra, model)
