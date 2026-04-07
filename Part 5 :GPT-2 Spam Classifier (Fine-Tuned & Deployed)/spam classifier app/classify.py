import torch
import tiktoken
import os
from GPT2_Archeticture import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

def load_classifier(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.out_head = torch.nn.Linear(768, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def classify_text(model, device, text, max_length=256, pad_token_id=50256):
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text = text.strip()
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    
    # Truncate if too long
    encoded = encoded[:max_length]
    
    encoded += [pad_token_id] * (max_length - len(encoded))
    
    input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        last_token_logits = logits[:, -1, :]
        probas = torch.softmax(last_token_logits, dim=-1)
        predicted_label = torch.argmax(probas, dim=-1).item()
        confidence = probas[0][predicted_label].item() * 100
        
    return "SPAM" if predicted_label == 1 else "HAM", confidence