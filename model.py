"""
model.py — LLM loader and hidden state extractor (fixed infrastructure, do not edit).

Loads a pre-trained DistilBERT model and provides utilities to extract hidden
states from all transformer layers for a list of input texts.

The model has 6 transformer layers plus an embedding layer, yielding 7 tensors
in ``outputs.hidden_states``. Layer indices 0–6 correspond to the embedding and
the 6 transformer layers respectively.

Students interact with the hidden states only through ``aggregation.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

_DEFAULT_MODEL = "distilbert-base-uncased"
_MAX_LENGTH = 128
_NUM_HIDDEN_LAYERS = 6  # DistilBERT has 6 transformer layers


def get_model_and_tokenizer(
    model_name: str = _DEFAULT_MODEL,
) -> tuple[DistilBertModel, DistilBertTokenizer]:
    """Load the pre-trained DistilBERT model and its tokenizer.

    The model is returned in evaluation mode with hidden state output enabled.

    Args:
        model_name: HuggingFace model identifier. Defaults to
                    ``"distilbert-base-uncased"``.

    Returns:
        A ``(model, tokenizer)`` tuple. The model is in eval mode.
    """
    print(f"[Model] Loading '{model_name}' ...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return model, tokenizer


def extract_hidden_states(
    model: DistilBertModel,
    tokenizer: DistilBertTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = _MAX_LENGTH,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract hidden states from all layers for a list of input texts.

    Processes texts in mini-batches and returns per-sample hidden states and
    attention masks. Hidden states from all 7 layers (embedding + 6
    transformer layers) are collected so that students can explore different
    layer choices in ``aggregation.py``.

    Args:
        model:      DistilBERT model (from ``get_model_and_tokenizer``).
        tokenizer:  Corresponding tokenizer.
        texts:      List of input strings to encode.
        device:     Device to run inference on.
        batch_size: Number of texts processed per forward pass.
        max_length: Maximum token sequence length (inputs are truncated/padded).

    Returns:
        A ``(all_hidden_states, all_attention_masks)`` tuple where:
          - ``all_hidden_states[i]`` is a tensor of shape
            ``(n_layers, seq_len, hidden_dim)`` for sample ``i``,
            where ``n_layers = 7`` (embedding + 6 transformer layers).
          - ``all_attention_masks[i]`` is a 1-D tensor of length ``seq_len``
            indicating real (1) vs. padding (0) tokens.

    Note:
        Hidden states are returned on CPU to conserve GPU memory.
    """
    model.to(device)
    model.eval()

    all_hidden_states: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="  Extracting hidden states",
        unit="batch",
        leave=False,
    ):
        batch_texts = texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # outputs.hidden_states is a tuple of 7 tensors:
        # index 0 → embedding layer, indices 1–6 → transformer layers.
        # Each tensor has shape (batch, seq_len, hidden_dim).
        hidden = torch.stack(outputs.hidden_states, dim=1)  # (batch, n_layers, seq_len, hidden_dim)
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            all_hidden_states.append(hidden[i].cpu())   # (n_layers, seq_len, hidden_dim)
            all_attention_masks.append(mask[i])          # (seq_len,)

    return all_hidden_states, all_attention_masks
