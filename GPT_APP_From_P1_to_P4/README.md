# GPT-2 Personal Assistant (Offline)

A fully offline **GPT-2 Personal Assistant** built from scratch using **PyTorch**.
The project loads a pretrained GPT-2 (124M) model, performs text generation with custom
sampling logic, and provides a clean desktop GUI for interaction.

No APIs. No internet. No cloud.
Everything runs locally.

---

## What This Project Does

- Loads a GPT-2 (124M) model using PyTorch
- Generates text token-by-token using:
  - Temperature sampling
  - Top-k filtering
- Runs completely offline
- Provides a modern desktop GUI
- Allows the user to control generation parameters in real time

---

## Key Features

- Offline AI assistant
- Custom GPT-2 architecture (not Hugging Face pipeline)
- Manual sampling implementation
- Adjustable generation controls:
  - Max new tokens
  - Temperature
  - Top-k
- Clean and professional GUI
- CPU & GPU support

---

## 📂 Project Files

- `Attention.py` – Multi-head attention module  
- `Transformer_block.py` – Transformer layer definitions  
- `GPT2_Architecture.py` – Full GPT-2 model implementation  
- `GPT_downloader.py` – Script to download GPT-2 pretrained weights  
- `Pretraining.py` – Load pretrained weights into the model  
- `generate.py` – Text generation logic  
- `GUI.py` – Desktop interface for the assistant  
- `README.md` – This file

## Requirements

Install dependencies once:

```bash
pip install torch tiktoken numpy customtkinter tensorflow requests tqdm
```

---




## Generation Parameters

- **Max New Tokens**  
  Controls how long the response is.

- **Temperature**  
  Controls randomness.
  - Low → more focused
  - High → more creative

- **Top-k**  
  Limits token choices to the top-k most likely tokens.

---

## Use Cases

- Personal assistant
- AI demo videos
- Educational projects
- LinkedIn technical showcase
- Understanding transformer inference internals

---

## Important Notes

- This project does NOT use external APIs.
- The model runs fully offline.
- Response quality depends on prompt design.
- This is a base GPT-2 model (not instruction-tuned).

---
## Next Steps

- Fine-tune for specific tasks

## Author

**Mohamed Waleed (ELMASRY)**  
AI / Machine Learning Engineer  

Built to understand transformers deeply — not just use them.
