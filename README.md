# Named Entity Recognition System

An end-to-end NER pipeline built on the CoNLL-2003 dataset — four architectures trained and
compared from scratch through to a fine-tuned Transformer, deployed as an interactive web app.

## 📌 Overview

This project builds a complete sequence-labeling pipeline that identifies and classifies named
entities — people, organizations, locations, and miscellaneous entities — in free text.

It covers the full workflow:
- Loading and exploring the CoNLL-2003 dataset (IOB tagging scheme)
- Preprocessing: tokenization, label encoding, padding, attention masks, token-label alignment
- Building and training **four architectures from scratch and via fine-tuning**
- Evaluating every model with entity-level precision/recall/F1 (`seqeval`)
- Deploying the best-performing model as an interactive Gradio web app

## 🧠 Models compared

| Model | Description |
|---|---|
| **LSTM** | Single-direction LSTM + pretrained GloVe embeddings |
| **BiLSTM** | Bidirectional LSTM, sees left and right context |
| **BiLSTM + CRF** | Adds a CRF layer to enforce valid label transitions and improve entity boundaries |
| **Fine-tuned DistilBERT** | HuggingFace Transformer fine-tuned for token classification |

### Results

| Model | Precision | Recall | F1 | Training time |
|---|---|---|---|---|
| LSTM | 0.7342790599195427 | 0.6140226628895185 | 0.6687879664448944 | 524.2186212539673 |
| BiLSTM | 0.7700759089093089 | 0.6825424929178471 | 0.7236718603341468 | 953.9223031997681 |
| BiLSTM + CRF | 0.8201047120418848 |0.693342776203966 | 0.7514151395951262 | 170.3873553276062 |
| Fine-tuned DistilBERT | 0.8823119777158774 | 0.8976266383280198 | 0.8899034240561896 | 206.28718972206116 |


## 📂 Repository structure

```
.
├── app.py                      # Gradio deployment app
├── requirements.txt            # Python dependencies for the app
└── NER_Project_Kaggle.ipynb    # Full training pipeline (run on Kaggle with GPU)
```

## 🚀 Getting started

### 1. Train (optional — a trained model is already included)
Open `NER_Project_Kaggle.ipynb` on [Kaggle](https://www.kaggle.com), enable a GPU accelerator,
and run all cells. It trains all four models, evaluates them, and exports the best one.

### 2. Run the app locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# venv\Scripts\activate           # Windows (CMD/PowerShell)
# source venv/bin/activate        # macOS/Linux

pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`), type a sentence, and
click **Predict** to see entities highlighted in real time.

## 🏷️ Dataset

[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) — a standard NER benchmark with
four entity types:

| Tag | Meaning |
|---|---|
| `PER` | Person |
| `ORG` | Organization |
| `LOC` | Location |
| `MISC` | Miscellaneous |
| `O` | Outside any entity |

## 🛠️ Tech stack

- **PyTorch** & **TensorFlow/Keras** — model architectures
- **HuggingFace Transformers** — tokenization and fine-tuning
- **seqeval** — entity-level evaluation metrics
- **Gradio** — interactive deployment
- **GloVe** — pretrained word embeddings for the LSTM-family models

## 📄 License

This project is released under the MIT License. Feel free to use and adapt it.

