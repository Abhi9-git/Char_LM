import Bigram_model
import pickle
import torch
import torch.nn.functional as F


with open("bigram_model.pkl", "rb") as f:
    checkpoint = pickle.load(f)

Bigram_model.embedding.load_state_dict(checkpoint["embedding"])

with torch.no_grad():
    Bigram_model.W.copy_(checkpoint["W"])

print("Parameters restored!")

def generate():
    out = []
    ix = 0  # Start token '.'

    while True:
        # Get embedding of current character
        x = Bigram_model.embedding(torch.tensor([ix]))

        # Predict next character
        logits = x @ Bigram_model.W
        probs = F.softmax(logits, dim=1)

        # Sample the next character
        ix = torch.multinomial(probs, num_samples=1).item()

        # Stop when end token is generated
        if ix == 0:
            break

        out.append(Bigram_model.itos[ix])

    return ''.join(out)

for _ in range(10):
    print(generate(),end='\t')
print()