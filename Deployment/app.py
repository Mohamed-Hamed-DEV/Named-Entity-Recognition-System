"""
NER System - Gradio Deployment App
-----------------------------------
Run locally (VS Code):
    1. python -m venv venv && source venv/bin/activate   (Windows: venv\\Scripts\\activate)
    2. pip install -r requirements.txt
    3. Unzip your trained model (downloaded from the Kaggle notebook) into ./ner_deployment_model/
       so this folder contains: config.json, model.safetensors (or pytorch_model.bin),
       tokenizer files, and label_maps.json
    4. python app.py
    5. Open the local URL Gradio prints (usually http://127.0.0.1:7860)
"""

import json
import os

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = os.environ.get("NER_MODEL_DIR", "./ner_deployment_model")

# Color map per entity type for the highlighted-text output
ENTITY_COLORS = {
    "PER": "#ff6b6b",
    "ORG": "#4dabf7",
    "LOC": "#51cf66",
    "MISC": "#ffd43b",
}

FRIENDLY_NAMES = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}


def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory '{MODEL_DIR}' not found. Unzip the model exported from the "
            f"Kaggle notebook into this folder first (see instructions at the top of app.py)."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    label_map_path = os.path.join(MODEL_DIR, "label_maps.json")
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            maps = json.load(f)
        id2label = {int(k): v for k, v in maps["id2label"].items()}
    else:
        id2label = model.config.id2label

    return tokenizer, model, id2label


tokenizer, model, id2label = load_model()


def iob_to_spans(tokens, tags):
    """Merge B-/I- IOB tags on whitespace-split tokens into (text, entity_type) spans
    suitable for gr.HighlightedText."""
    spans = []
    current_words, current_type = [], None

    def flush():
        nonlocal current_words, current_type
        if current_words:
            spans.append((" ".join(current_words), current_type))
            current_words, current_type = [], None

    for tok, tag in zip(tokens, tags):
        if tag == "O":
            flush()
            spans.append((tok, None))
        elif tag.startswith("B-"):
            flush()
            current_words = [tok]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_words.append(tok)
        else:  # I- without a matching B- (defensive) -> start new span
            flush()
            current_words = [tok]
            current_type = tag[2:] if "-" in tag else tag
    flush()
    return spans


def predict(sentence):
    if not sentence or not sentence.strip():
        return {"text": "", "entities": []}, []

    tokens = sentence.strip().split()
    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**encoding).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = encoding.word_ids()

    tags = []
    prev_word = None
    for idx, wid in zip(pred_ids, word_ids):
        if wid is None or wid == prev_word:
            continue
        tags.append(id2label[idx])
        prev_word = wid

    spans = iob_to_spans(tokens, tags)

    # Table of detected entities for a clean summary view
    table = [
        [text, FRIENDLY_NAMES.get(etype, etype)]
        for text, etype in spans
        if etype is not None
    ]
    if not table:
        table = [["No entities detected", ""]]

    return spans, table


with gr.Blocks(title="NER System") as demo:
    gr.Markdown("# 🏷️ Named Entity Recognition\nEnter a sentence and click **Predict** to see detected entities highlighted in real time.")

    with gr.Row():
        inp = gr.Textbox(label="Your sentence", placeholder="e.g. Apple hired John in New York.", lines=2)

    btn = gr.Button("Predict", variant="primary")

    highlighted = gr.HighlightedText(
        label="Detected entities",
        color_map=ENTITY_COLORS,
        show_legend=True,
    )
    table = gr.Dataframe(headers=["Entity text", "Type"], label="Entity summary")

    btn.click(fn=predict, inputs=inp, outputs=[highlighted, table])
    inp.submit(fn=predict, inputs=inp, outputs=[highlighted, table])

    gr.Examples(
        examples=[
            "Apple hired John in New York.",
            "Barack Obama visited Berlin last July.",
            "Elon Musk founded SpaceX and later acquired Twitter.",
        ],
        inputs=inp,
    )

if __name__ == "__main__":
    demo.launch()
