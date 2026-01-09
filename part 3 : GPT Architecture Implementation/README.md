# LLM From Scratch – GPT Architecture Implementation

> ⚠️ This implementation focuses on understanding, not performance or production use.

---

## 🧠 Goal of This Stage

After implementing:
- Text preprocessing & tokenization
- Embeddings & positional encodings
- Self-attention and multi-head attention

the next step is assembling all components into a **complete GPT-style model** capable of **autoregressive next-token prediction**.

This notebook shows **how real GPT models are constructed internally**, block by block.

---

## 📌 Implemented Components

### 1️⃣ GPT Model Skeleton
- Token embedding layer
- Positional embedding layer
- Stack of Transformer blocks
- Final linear layer projecting to vocabulary size

---

### 2️⃣ Layer Normalization (From Scratch)
- Manual implementation of LayerNorm
- Learnable scale (γ) and bias (β)
- Explanation of why LayerNorm stabilizes deep Transformers

---

### 3️⃣ Feed-Forward Network (FFN)
- Position-wise MLP applied independently to each token
- Architecture:
  ```
  Linear → GELU → Linear
  ```
- Expands and contracts embedding dimension

---

### 4️⃣ GELU Activation Function
- Smooth, probabilistic gating behavior
- Used instead of ReLU in GPT models
- Improves optimization and gradient flow

---

### 5️⃣ Residual Connections
- Skip connections around:
  - Self-attention
  - Feed-forward network
- Prevents vanishing gradients
- Enables deep model stacking

---

### 6️⃣ Transformer Block (GPT Block)
Each block contains:
- Layer Normalization
- Causal Multi-Head Self-Attention
- Residual connection
- Feed-Forward Network
- Second residual connection

This matches the structure used in real GPT architectures.

---

### 7️⃣ Full GPT Model
- Multiple stacked Transformer blocks
- Final LayerNorm
- Output logits for next-token prediction

---

### 8️⃣ Parameter Counting
- Computes total trainable parameters
- Helps understand model scaling behavior

---

## 🧩 Relation to the Full Project

This notebook integrates directly with:
- Tokenization & embeddings (Stage 1)
- Attention mechanisms (Stage 2)

Together, these stages form a **complete forward pass of a GPT-style Large Language Model**.

---

## ⚠️ Important Notes
- Decoder-only architecture (no encoder)
- Autoregressive causal masking
- No high-level Transformer APIs
- Code written explicitly for learning

---

## 🚀 Next Planned Steps
- Training loop implementation
- Cross-entropy loss
- Autoregressive text generation
- Sampling strategies (greedy, temperature, top-k)

---

## 📚 Educational Disclaimer

This project is built strictly for **educational purposes** to deeply understand how GPT-style Large Language Models are implemented internally.
