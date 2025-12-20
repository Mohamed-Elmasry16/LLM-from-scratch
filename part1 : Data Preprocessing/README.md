
# LLM From Scratch – Text Preprocessing & Tokenization

This repository contains an **educational implementation of text preprocessing and tokenization**, built from scratch as the **first stage of a larger project to build a Large Language Model (LLM) from first principles**.

> ⚠️ **Note:** The purpose of this code is **deep understanding**, not performance or production use.

---

## 🧠 Project Goal

Modern Large Language Models cannot process raw text directly.  
Text must first be transformed into numerical representations through several preprocessing steps.

This notebook implements and explains those steps **explicitly and manually**, avoiding high-level NLP libraries to build intuition about how LLMs work internally.

---

## 📌 Implemented Features

### 1️⃣ Rule-Based Tokenization
- Splits raw text using regular expressions
- Handles:
  - Words
  - Punctuation
  - Whitespace cleanup
- Serves as an introductory tokenizer to explain the core idea of text segmentation

### 2️⃣ Vocabulary Construction
- Extracts all unique tokens from the corpus
- Sorts tokens to ensure deterministic indexing
- Builds bidirectional mappings:
  ```text
  token → integer ID
  integer ID → token
  ```
- This vocabulary is the foundation for all later stages

### 3️⃣ Encoding & Decoding
- **Encoding:** Converts text into sequences of token IDs  
- **Decoding:** Converts token IDs back into readable text  
- Used to verify correctness of the tokenizer and vocabulary

### 4️⃣ Special Tokens Handling
- Adds special tokens to support language model training:
  - `<PAD>` – padding
  - `<UNK>` – unknown tokens
  - `<BOS>` – beginning of sequence
  - `<EOS>` – end of sequence
- These tokens are essential for:
  - Batch processing
  - Sequence alignment
  - Stable training behavior

### 5️⃣ Byte Pair Encoding (BPE)
- Simplified implementation of subword tokenization
- Key concepts covered:
  - Character-level initialization
  - Iterative pair frequency counting
  - Pair merging
  - Vocabulary compression
- Provides better handling of rare and unseen words  
- Reflects the core idea behind tokenizers used in modern LLMs (e.g., GPT)

### 6️⃣ Preparing Training Sequences (Next-Token Prediction)
- Demonstrates autoregressive language model training
- Given a token sequence:
  ```text
  x₁ x₂ x₃ x₄
  ```
  The model learns:
  ```text
  Input : x₁ x₂ x₃
  Target: x₂ x₃ x₄
  ```
- This sliding-window approach is fundamental to LLM training

### 7️⃣ Custom Dataset & Data Loader
- Builds input–target pairs manually
- Supports batching
- Mimics a deep learning data pipeline
- Prepares data for direct use in a training loop

### 8️⃣ Token Embeddings
- Maps token IDs to dense vectors
- Implements an embedding matrix
- Converts discrete tokens into continuous representations suitable for neural networks

### 9️⃣ Positional Embeddings
- Since transformers have no inherent sense of order:
  - Absolute positional embeddings are added
  - Token embeddings and positional embeddings are combined:
    ```text
    final_embedding = token_embedding + positional_embedding
    ```
- Produces the final input representation required by self-attention

---

## 🧩 Relation to LLM Architecture
This preprocessing pipeline fully prepares data for the next stages of an LLM, including:
- Self-Attention
- Multi-Head Attention
- Transformer Blocks
- Autoregressive Text Generation

Without this preprocessing, an LLM cannot be trained.

---

## ⚠️ Important Notes
- Not production-ready
- No optimization or heavy abstraction is used
- Focus is on clarity, transparency, and learning

---

## 🚀 Planned Next Steps
- Scaled Dot-Product Attention
- Multi-Head Attention
- Transformer Block implementation
- Training loop from scratch
- Text generation

---

## 📚 Educational Disclaimer
This repository is built strictly for learning purposes to understand how Large Language Models work internally, step by step.
