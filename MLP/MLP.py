# 3 layer perceptron for character level language model
import torch
import os
import torch.nn.functional as F
import pickle

words = open("../data/shakespeare_100k.txt").read().splitlines()

lookup=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(lookup)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

window_size = 5
def build_dataset(words):
    X, Y =[], []

    for w in words: 
        context = [0]*window_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:]+[ix] # crop and append

    X=torch.tensor(X)
    Y=torch.tensor(Y)

    return X, Y
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xts, Yts = build_dataset(words[n2:])



vocab_size = len(stoi)
embedding_dim = 5

C = torch.randn((vocab_size, embedding_dim))
W1 = torch.randn((window_size * embedding_dim, 300))
b1 = torch.randn(300)

W2 = torch.randn((300, 100))
b2 = torch.randn(100)

W3 = torch.randn((100, vocab_size))
b3 = torch.randn(vocab_size)

parameters = [C, W1, b1, W2, b2, W3, b3]

# print(sum(p.nelement() for p in parameters) + 'parameters')


CHECKPOINT_PATH = "checkpoint.pth"
PICKLE_PATH = "MLP.pkl"


def save_checkpoint(iteration):
    checkpoint = {
        "iteration": iteration,
        "C": C.detach().clone(),
        "W1": W1.detach().clone(),
        "b1": b1.detach().clone(),
        "W2": W2.detach().clone(),
        "b2": b2.detach().clone(),
        "W3": W3.detach().clone(),
        "b3": b3.detach().clone()
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at iteration {iteration}")


def load_checkpoint():

    if os.path.exists(CHECKPOINT_PATH):

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True
        )

        for name, p in zip(
            ["C", "W1", "b1", "W2", "b2", "W3", "b3"],
            parameters
        ):
            p.data.copy_(checkpoint[name])

        start_iteration = checkpoint["iteration"]

        print(f"Resuming from iteration {start_iteration}")

        return start_iteration

    print("No checkpoint found. Starting fresh.")
    return 0

def export_pickle():

    model = {
        "C": C.detach().clone(),
        "W1": W1.detach().clone(),
        "b1": b1.detach().clone(),
        "W2": W2.detach().clone(),
        "b2": b2.detach().clone(),
        "W3": W3.detach().clone(),
        "b3": b3.detach().clone(),
        "stoi": stoi,
        "itos": itos,
        "window_size": window_size,
        "embedding_dim": embedding_dim
    }

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model exported to {PICKLE_PATH}")

def train(total_iterations=100):

    for p in parameters:
        p.requires_grad = True

    start_iteration = load_checkpoint()

    for i in range(total_iterations):

        # Forward pass
        emb = C[Xtr]
        h1 = torch.tanh(emb.view(-1, 25) @ W1 + b1)
        h2 = torch.relu(h1 @ W2 + b2)
        logits = h2 @ W3 + b3

        loss = F.cross_entropy(logits, Ytr)

        # Backward pass
        for p in parameters:
            p.grad = None

        loss.backward()

        # Update
        with torch.no_grad():
            for p in parameters:
                p -= 0.1 * p.grad


        print(f"Iteration: {i+1}/{total_iterations} | Loss: {loss.item():.4f}")
        if (i + 1) % 50 == 0:
            save_checkpoint(i + 1)

    print(f"\nTraining completed. Final loss: {loss.item():.4f}")

    choice = input("\nEnter 's' to export model as pickle, or press Enter to skip: ") # save the model if weights are desirable

    if choice.lower() == 's':
        export_pickle()


if __name__ == "__main__":
    q=int(input('enter training iterations '))
    train(total_iterations=q)