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
```


### 2. Polyglot Language Setup

```bash
polyglot download embeddings2.hi
polyglot download ner2.hi
```

### 🧠 Model Architecture
- Tokenizer: Polyglot (for Hindi) → mapped to mBERT vocab.

- Model: bert-base-multilingual-cased fine-tuned for classification.

### 🧾 Label Mapping

```bash
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
```

### 🚀 Training Configuration
- Model: mBERT (bert-base-multilingual-cased)

- Epochs: 60 (with early stopping)

- Batch Size: 22

- Max Sequence Length: 128

- Evaluation Strategy: Per epoch

- EarlyStopping: Stops training if f1 does not improve after 10 epochs

### 🧮 Metrics Computed
- Accuracy

- F1 Score (Weighted)

- Precision (Weighted)

- Recall (Weighted)

## 🧠 Inference Example

> To run predictions on new Hindi text:

```bash
def classify_text(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
    prediction = torch.argmax(outputs.logits, dim=1)
    return list(label_mapping.keys())[prediction.item()]

```

