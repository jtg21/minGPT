import random
from unittest import result
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def give_exam(ndigit, model, trainer, dataset, batch_size=32, max_batches=-1):
    torch.set_printoptions(edgeitems=5)
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        d1d2 = x[:, :ndigit*2]
        d1d2d3 = sample(model, d1d2, ndigit+1)
        d3 = d1d2d3[:, -(ndigit+1):]
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        # decode the integers from individual digits
        d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
        d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i
        correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line, lol
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            judge = 'YEP!!!' if correct[i] else 'NOPE'
            if not correct[i]:
                print("GPT claims that %03d + %03d = %03d (gt is %03d; %s)" 
                      % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i], judge))
        
        if max_batches >= 0 and b+1 >= max_batches:
            break

    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))

def give_dataset_exam_sq(ndigit, nextra, model, trainer, dataset, batch_size=32, max_batches=-1):
    torch.set_printoptions(edgeitems=5)
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        d1 = x[:, :ndigit]
        pred = sample(model, d1, nextra+1)
        pred = pred[: ,ndigit+1:nextra+ndigit+1]
        y = y[:, ndigit:]
        correct = (pred == y).cpu()
        for i in range(x.size(0)):
            results.append(torch.all(correct[i]))
                
            judge = 'YEP!!!' if torch.all(correct[i]) else 'NOPE'
            if not torch.all(correct[i]):
                print(f'{judge} - Prediction: {pred[i]} Target: {y[i]}')
                pass
    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))

def give_random_exam_sq(ndigit, nextra, model):
    test_size = 100
    correct = 0
    for _ in range(test_size):
        test = np.random.randint(10**ndigit, size=1)
        sq_test = test ** 0.5

        render = f"%0{ndigit}d" % (test)
        test = [int(d) for d in render]
        context = torch.tensor(test, dtype=torch.long).unsqueeze(0)
        
        render = str(sq_test[0])[:nextra+1]

        expected = [int(d) for d in render if d != '.']

        expected += [0] if len(expected) < nextra else []

        if expected[1:] == [0, 0]:
            expected = [0] + [expected[0]] + [0]
        
        expected = torch.tensor(expected, dtype=torch.long)

        y = sample(model, context, nextra+1, temperature=1.0, sample=True, top_k=5)[0]
        prediction = y[ndigit+1:]

        if torch.all(torch.eq(expected, prediction)):
            correct += 1
        else:
            print(f"Test: {test} Prediction: {prediction} Expected: {expected}")
    print(f'Correct: {correct}/{test_size}')
        
