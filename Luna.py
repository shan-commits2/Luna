from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ---------------- Vocabulary ---------------- #
vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?():=+-*/'\"\n")
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(text): 
    return [stoi[c] for c in text.lower() if c in stoi]

def decode(tokens): 
    return "".join([itos[t] for t in tokens])

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
model_path = "tinygpt.pt"

# Load existing model if available
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Loaded existing model.")

model.eval()  # set model to evaluation mode

# ---------------- Generation Function ---------------- #
def generate(prompt, length=200, temperature=0.8):
    tokens = encode(prompt)
    tokens = torch.tensor([tokens])
    hidden = None
    for _ in range(length):
        out, hidden = model(tokens[:, -1:], hidden)
        probs = torch.softmax(out[:, -1] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())

# ---------------- Flask App ---------------- #
app = Flask(__name__)

@app.route("/")
def home():
    return "TinyGPT Flask API is running!"

@app.route("/generate", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        data = request.get_json()
        prompt = data.get("prompt", "")
    else:
        prompt = request.args.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    response = generate(prompt, length=120, temperature=0.9)
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()

    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
