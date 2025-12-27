# LLM From Scratch – Attention Mechanism

This repository contains an **educational implementation of the Attention mechanism**, built as the **second stage of a larger project to build a Large Language Model (LLM) from first principles**.

> ⚠️ **Note:** The goal of this implementation is **deep understanding**, not efficiency or production use.

---

## 🧠 Project Goal

After text is converted into embeddings (Part 1), a model must learn how to **dynamically relate tokens to each other** within a sequence.

This notebook implements the **Attention mechanism step by step**, explicitly showing how modern Transformer-based LLMs compute context-aware representations.

---

## 📌 Implemented Features

### 1️⃣ Attention Intuition
- Explains why simple embeddings are insufficient  
- Shows how tokens assign importance to other tokens  
- Motivates the need for attention in sequence modeling  

---

### 2️⃣ Self-Attention (Conceptual Form)
- Computes similarity between token embeddings  
- Uses dot-product attention  
- Applies softmax to obtain attention weights  
- Produces weighted combinations of token vectors  

This step builds intuition **before introducing Q, K, and V**.

---

### 3️⃣ Scaled Dot-Product Attention
- Introduces Query (Q), Key (K), and Value (V)  
- Implements the standard attention formula:
  
  Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

- Explains why scaling by √dₖ is necessary for stable training  

---

### 4️⃣ Causal (Masked) Self-Attention
- Implements autoregressive masking  
- Ensures tokens can only attend to previous positions  
- Prevents information leakage during training  

---

### 5️⃣ Batched Attention Computation
- Extends attention to batched inputs  
- Clearly tracks tensor shapes:
  
  (batch_size, sequence_length, embedding_dim)

---

### 6️⃣ Multi-Head Attention
- Splits embeddings into multiple attention heads  
- Each head learns different token relationships  
- Includes linear projections, parallel attention, concatenation, and output projection  

---

## 🧩 Relation to LLM Architecture

This attention pipeline forms the **core computation** inside a Transformer block and prepares representations for:
- Feed-forward networks  
- Residual connections  
- Layer normalization  

---

## ⚠️ Important Notes
- Not optimized for speed or memory  
- No use of high-level Transformer libraries  
- All steps are written explicitly for learning purposes  

---

## 🚀 Planned Next Steps
- Transformer block implementation  
- Residual connections and layer normalization  
- Decoder-only Transformer  
- Training loop from scratch  
- Autoregressive text generation  

---

## 📚 Educational Disclaimer

This repository is built strictly for **educational purposes**, aiming to provide a transparent, step-by-step understanding of how attention works inside Large Language Models.

