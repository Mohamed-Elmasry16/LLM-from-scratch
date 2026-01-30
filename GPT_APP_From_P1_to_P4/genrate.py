# genrate.py
import os
import torch
from GPT2_Archeticture import GPTModel
import tiktoken

# ================================
# 1️⃣ Set the device (GPU if available)
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# 2️⃣ GPT-2 Model Configuration
# ================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # number of tokens in GPT-2 vocabulary
    "context_length": 1024,    # maximum sequence length
    "emb_dim": 768,            # embedding size
    "n_heads": 12,             # number of attention heads
    "n_layers": 12,            # number of transformer blocks
    "drop_rate": 0.1,          # dropout probability
    "qkv_bias": True            # use bias in Q, K, V projections
}

# Path to the saved model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder
MODEL_PATH = os.path.join(BASE_DIR, "GPT2_124M_torch.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ================================
# 3️⃣ Load the model once
# ================================
GPT = GPTModel(GPT_CONFIG_124M)
GPT.load_state_dict(torch.load(MODEL_PATH, map_location=device))
GPT.to(device)
GPT.eval()  # put the model in evaluation mode

# ================================
# 4️⃣ Set up the tokenizer
# ================================
tokenizer = tiktoken.get_encoding("gpt2")

# ================================
# 5️⃣ Function to generate token IDs
# ================================
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate new tokens from the model given a starting sequence (idx)
    """
    for _ in range(max_new_tokens):
        # only use the last 'context_size' tokens for context
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # only look at the last token

        # filter logits to only consider top-k options
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # apply temperature to control randomness
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from probabilities
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # take most likely token

        # stop if end-of-sequence token is reached
        if eos_id is not None and idx_next == eos_id:
            break

        # append new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ================================
# 6️⃣ Helper functions to convert text <-> tokens
# ================================
def text_to_token_ids(text):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension

def token_ids_to_text(token_ids):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

# ================================
# 7️⃣ Main function to generate text from a prompt
# ================================
def generate_text(prompt, max_new_tokens=50, temperature=0.7, top_k=25):
    """
    This is the function you can call from GUI or scripts.
    It takes your prompt, generates new text, and returns it as a string.
    """
    idx = text_to_token_ids(prompt).to(device)
    generated_ids = generate(
        GPT, 
        idx, 
        max_new_tokens, 
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=temperature, 
        top_k=top_k
    )
    return token_ids_to_text(generated_ids)
