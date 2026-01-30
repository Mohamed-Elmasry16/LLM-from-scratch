import os
import glob
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tiktoken

from GPT_downloader import download_and_load_gpt2
from GPT2_Archeticture import GPTModel

# =========================
# Paths and device setup
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Download/load GPT-2
# =========================
settings, params = download_and_load_gpt2(model_size="124M", models_dir=MODELS_DIR)

# =========================
# Functions to assign weights
# =========================
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right, dtype=left.dtype, device=left.device))


def load_weights_into_gpt(gpt, params):
    # Token and positional embeddings
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])

    # Transformer blocks
    for b in range(len(params["blocks"])):
        # Attention QKV
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)

        gpt.trf_blocks[b].att.W_Q.weight = assign(gpt.trf_blocks[b].att.W_Q.weight, q_w.T)
        gpt.trf_blocks[b].att.W_K.weight = assign(gpt.trf_blocks[b].att.W_K.weight, k_w.T)
        gpt.trf_blocks[b].att.W_V.weight = assign(gpt.trf_blocks[b].att.W_V.weight, v_w.T)

        gpt.trf_blocks[b].att.W_Q.bias = assign(gpt.trf_blocks[b].att.W_Q.bias, q_b)
        gpt.trf_blocks[b].att.W_K.bias = assign(gpt.trf_blocks[b].att.W_K.bias, k_b)
        gpt.trf_blocks[b].att.W_V.bias = assign(gpt.trf_blocks[b].att.W_V.bias, v_b)

        # Attention output projection
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # Feedforward layers
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        # Layer norms
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    # Final output
    gpt.out_head.weight = gpt.tok_emb.weight
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

# =========================
# Model initialization
# =========================
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT = GPTModel(GPT_CONFIG_124M)
GPT.eval()
load_weights_into_gpt(GPT, params)
GPT.to(device)

print("✅ GPT-2 124M model loaded successfully!")

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "GPT2_124M_torch.pth")
torch.save(GPT.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

