import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# -------


# reading files
with open('./tiny-shakespeare.txt', 'r') as f:
    text = f.read()


# building vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)


# mapping characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# lambda function to convert text to integers
def encode(x): return [stoi[ch] for ch in x]
def decode(x): return ''.join([itos[ch] for ch in x])


# encoding entire text and storing in torch tensor
data = torch.tensor(encode(text), dtype=torch.int64)


# test train split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# batch generator
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ A single attention head. """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(head_size, head_size, bias=False)
        self.query = nn.Linear(head_size, head_size, bias=False)
        self.value = nn.Linear(head_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # dot product of query and key
        # (B, T, 16) @ (B, 16, T) = (B, T, T)
        wei = q @ k.transpose(-2, -1) * C ** -0.5

        tril = torch.tril(torch.ones(T, T))
        # these will have actual weight as future iterations have interests in past in varying degrees
        # wei = torch.zeros((T, T))
        # we dont really care about future
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """ Multi-head attention. """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        return out


class BigramModel(nn.Module):
    """Bigram model for language modeling."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = MultiHeadAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # Batch, Time, Channel
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # Time, Channel
        x = tok_emb + pos_emb  # Batch, Time, Channel
        x = self.sa_head(x)  # Batch, Time, Channel
        logits = self.lm_head(x)  # Batch, Time, Vocabulary

        if target is None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)

            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Batch, Block
            # get prediction for next token
            logits, loss = self(idx_cond)  # Batch, Time, Channel
            # focus only on last time step
            logits = logits[:, -1, :]  # Batch, Channel
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append to the input
            idx = torch.cat([idx, idx_next], dim=1)  # Batch, Time + 1

        return idx


# loss estimator
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for splt in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(splt)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[splt] = losses.mean()
    model.train()
    return out


model = BigramModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    # evaluate loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"iter {iter}, train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

    # sample batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
idx = torch.zeros((1, 1), dtype=torch.int64, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
