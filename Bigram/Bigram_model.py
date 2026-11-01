import torch
import os
import pickle
import torch.nn as nn
import torch.nn.functional as F

words= open("../data/shakespeare_100k.txt", "r").read().splitlines()

lookup=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(lookup)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}


vocab_size = len(stoi)
embedding_dim = 8

embedding = nn.Embedding(vocab_size, embedding_dim)
W = torch.randn(embedding_dim, vocab_size, requires_grad=True)

if os.path.exists('bigram_model.pth'):
    checkpoint = torch.load('bigram_model.pth', weights_only=True)

    embedding.load_state_dict(checkpoint['embedding'])
    W = checkpoint['W'].clone().detach().requires_grad_(True)

    print("Previous weights restored!")
else:
    print("Starting with fresh weights.")

parameters = list(embedding.parameters()) + [W]

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)


learning_rate = 2
epochs = 100

def train():
    for epoch in range(epochs):

        # Forward pass
        X = embedding(xs)
        logits = X @ W

        loss = F.cross_entropy(logits, ys)

        # Reset gradients
        for p in parameters:
            p.grad = None

        # Backward pass
        loss.backward()

        # Update parameters
        with torch.no_grad():
            for p in parameters:
                p -= learning_rate * p.grad

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    torch.save({
        'embedding': embedding.state_dict(),
        'W': W.detach()
    }, 'bigram_model.pth')

    print("Parameters saved successfully!")

    def save_model_file():
        with open("bigram_model.pkl", "wb") as f:
            pickle.dump({
                "embedding": embedding.state_dict(),
                "W": W.detach()
            }, f)

        print("Parameters saved!")
    x=input("if you want to save the model enter s:  ")
    if x=='s':
        save_model_file()



if __name__ == "__main__":
    train()