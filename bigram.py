import os
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
batch_size = 32 # 
block_size = 8 #
max_iters = 3000

eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
# -------------------

torch.manual_seed(1337)

# Read and inspect the data:
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
      
# Character list - unique characters that occur in this text:
chars = sorted(list(set(text))) # Sort a list of characters that appear in the text: [, . & .' etc]
vocab_size = len(chars)

# Tokenize: mapping (look-up table) from characters to integers:
# character to integer and viceversa
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

# Encode 
encode = lambda s: [stoi[c] for c in s] # takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # takes a list of integers, outputs a string

# Train and test splits:
# Take all the text in tiny shakespeare, encode it and wrap it into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long) # dtype int
# First 90% will be train:
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# Data loading:
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # Data array
    # Generate random positions to grab a chunk out of -- generate batch_size number of random offsets. 
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 numbers (batch_size = 4), randomly generated between 0 and len(data)-block_size
    x = torch.stack([data[i:i+block_size] for i in ix]) # First block_size characters starting at 'i'
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Offset by one of x
    # We get these chunks for every one integer 'i' in ix. Stack 1D tensors as rows: 4x8 tensor. 
    x, y = x.to(device), y.to(device) # move the data to device
    return x, y

@torch.no_grad() # Context manager: we don't call .backward() (backprop) on here -- more efficient. 
def estimate_loss():
    # Avg the loss over multiple batches
    # Iterate over eval_iters, average the loss, and get the loss for both splits (train, eval). 
    out = {}
    
    # So far, our model behaves the same for training and evaluation modes.
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
        
# Super simple Bigram model:
class BigramLanguageModel(nn.Module):
    # Subclass of nn.module 
    # Very simple model, tokens are not talking to each other
    
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logitd for the next token from a look-up table
        # Create a token embedding table of size (vocab size, vocab size): 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # inputs and targets are passed to the token embedding table
        # e.g., number 24 in xb will plug out the 24th row of the embedding table, number 43 will do the same for 43rd row, etc. 
        # idx and targets are both (B, T) tensor of integers
        # (Batch, Time, Channel) -- (batch_size, block_size, vocab_size) = (4, 8, 64)
        logits = self.token_embedding_table(idx) # (B, T, C) -- Scores for the next character of the sequence 
        # -- we're predicting based on the identity of a single token
        
        if targets is None:
            loss = None
        else:
            # Good way to measure the loss or the quality of the predictions is to use the negative log-likelohood (cross-entropy) 
            # Torch expects the channel to be the 2nd dimension -> (B, C, T), so reshape
            B, T, C = logits.shape # (4, 8, 64)
            logits = logits.view(B*T, C) # (4x8, 64)
            # Targets are (B, T):
            targets = targets.view(B*T) # or (-1), size (4x9)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is the current context of some characters in some batch
        # idx is (B, T) array -- e.g., (4, 8)
        
        # Note that this function gets fed with the full idx to predict a single character at a time. 
        # it is only necessary, for predicting the next character, to know the previous one.
        # in line logits[:, -1, :], we're only using the last one instead of the whole history. 
        # so far, no needed. later history will be used. 
        
        # Generate data from the model
        for _ in range(max_new_tokens):
            # Get te predictions for the current idx
            logits, loss = self(idx)
            # Focus only on the last time step
            # logits are (B, T, C) --> convert them to (B, C) 
            # taking the last time-step, as those are the predictions of what comes next.
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the probability distribution, give 1 sample (integer)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), for each of the batch dimension, 
            # we get a single prediction of what comes next. 
            
            # Append sampled index to the running sequence: 
            # whatever is predicted, is concatenated on top of the previous idx along 
            # the first dimension (time dimension) to get (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) e.g., -- (4,9) 
        return idx

# Call the model
model = BigramLanguageModel(vocab_size)
m = model.to(device) # move the model to device
    
# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # takes parameters and updates the gradients

# Optimize:
for iter in range(max_iters):
    
    # Every once in a while eval the loss on train and val sets:
    if iter % eval_interval==0:
        losses = estimate_loss() # averages the loss over multiple batches 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss{losses['val']:.4f}")
        
    # Sample a batch of data from the data loader
    xb, yb = get_batch('train')
    
    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # Zeroing the gradients from the previous step 
    loss.backward() # get the gradients from all the parameters
    optimizer.step() # use the gradients to update 
    
# Generate from the model:
context = torch.zeros((1,1), dtype=torch.long, device=device) # Create it on the device
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))