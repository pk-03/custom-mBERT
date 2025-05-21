# custom-mBERT

# Hindi Sentiment Classification using Polyglot Tokenization + mBERT

This project fine-tunes the `bert-base-multilingual-cased` (mBERT) model for Hindi sentiment classification. It uses **Polyglot** for tokenization and adapts the output to be compatible with mBERT's vocabulary.

---

## 📁 Dataset Format

Ensure your dataset is a Pandas DataFrame with the following columns:

- `Reviews` — Hindi text (input).
- `labels` — Sentiment label (`positive`, `neutral`, or `negative`).

---

## 🧪 Setup Instructions

### 1. Install Dependencies

```bash
pip install torch transformers pandas scikit-learn polyglot pyicu pycld2 morfessor
