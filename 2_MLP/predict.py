
import torch
import torch.nn.functional as F
import pickle


# Load trained model
with open("MLP.pkl", "rb") as f:
    model = pickle.load(f)


# Restore parameters
C = model["C"]
W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]
W3 = model["W3"]
b3 = model["b3"]

stoi = model["stoi"]
itos = model["itos"]

window_size = model["window_size"]


def generate():

    context = [0] * window_size
    output = ""

    with torch.no_grad():

        while True:

            # Forward pass
            emb = C[torch.tensor([context])]

            h1 = torch.tanh(emb.view(1, -1) @ W1 + b1)
            h2 = torch.relu(h1 @ W2 + b2)

            logits = h2 @ W3 + b3

            # Convert logits into probabilities
            probs = F.softmax(logits, dim=1)

            # Sample next character
            ix = torch.multinomial(probs, num_samples=1).item()

            # Stop if end-of-word character is generated
            if ix == 0:
                break

            output += itos[ix]

            # Update context
            context = context[1:] + [ix]

    return output


# predict 10 words
print("\nGenerated Words:\n")

for i in range(20):
    word = generate()
    print(f"{i+1}. {word}")
