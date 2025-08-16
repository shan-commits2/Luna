import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# ---------------- Vocabulary ---------------- #
vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?():=+-*/'\"\n")
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(text): return [stoi[c] for c in text.lower() if c in stoi]
def decode(tokens): return "".join([itos[t] for t in tokens])

# ---------------- Model ---------------- #
class TinyGPT(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256, layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

model = TinyGPT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)

# ---------------- Training Data ---------------- #
training_data = """
AI: I must always be kind and helpful.
AI: I cannot generate inappropriate content.
Human: hello
AI: Hello! How are you today?

Human: write a python script that prints hello world
AI: Sure! Here is a simple Python script:

print("Hello, world!")

Human: how to add two numbers in python?
AI: You can do it like this:

a = 5
b = 3
print(a + b)

Human: create a for loop that prints numbers 1 to 5
AI: Here's an example:

for i in range(1, 6):
    print(i)

Human: tell me a joke
AI: Why did the computer get cold? Because it forgot to close its Windows!

Human: what is AI?
AI: AI stands for Artificial Intelligence, which means making machines think and learn like humans.

Human: bye
AI: Goodbye! Have a great day!
"""

data = encode(training_data)

# Chunk training data into sequences
seq_len = 64
inputs, targets = [], []
for i in range(len(data) - seq_len):
    inputs.append(data[i:i+seq_len])
    targets.append(data[i+1:i+1+seq_len])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

# ---------------- Training Loop ---------------- #
epochs = 800
batch_size = 32

if os.path.exists("tinygpt.pt"):
    model.load_state_dict(torch.load("tinygpt.pt"))
    print("Loaded existing model.")
else:
    print("Training model from scratch...")
    for epoch in range(epochs):
        # mini-batch training
        for i in range(0, len(inputs), batch_size):
            xb = inputs[i:i+batch_size]
            yb = targets[i:i+batch_size]

            optimizer.zero_grad()
            out, _ = model(xb)
            loss = criterion(out.view(-1, vocab_size), yb.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    torch.save(model.state_dict(), "tinygpt.pt")
    print("Model saved.")

# ---------------- Generation ---------------- #
def generate(prompt, length=200, temperature=0.8):
    model.eval()
    tokens = encode(prompt)
    tokens = torch.tensor([tokens])
    hidden = None
    for _ in range(length):
        out, hidden = model(tokens[:, -1:], hidden)
        probs = torch.softmax(out[:, -1] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())

# ---------------- Interactive Chat ---------------- #
print("\n--- Chat Mode ---")
context = "The following is a conversation between Human and AI.\n"
while True:
    user_input = input("me: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("AI: Goodbye! 👋")
        break

    context += f"Human: {user_input}\nAI:"
    response = generate(context, length=120, temperature=0.9)

    # extract only AI response after last "AI:"
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()

    print("AI:", response)
    context += response + "\n"
