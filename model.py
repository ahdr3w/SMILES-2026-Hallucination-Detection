"""
model.py — LLM loader and hidden state extractor (fixed infrastructure, do not edit).

Loads ``Qwen/Qwen2.5-0.5B`` and provides utilities to extract hidden states
from all transformer layers for a list of input texts.

Qwen2.5-0.5B is a **decoder-only** (causal) language model.  Its internal
representations are extracted by running a forward pass with the full
``prompt + generated_response`` text and collecting the hidden states from
every transformer layer.  Unlike encoder models, the most informative position
in a causal LM is typically the **last real (non-padding) token**, as it
attends to the entire preceding context.

Architecture reference (``model.config``)
------------------------------------------
  ``model.config.num_hidden_layers``  — number of transformer layers (24)
  ``model.config.hidden_size``        — hidden dimension per token  (896)
  Hidden-state index 0 → token embeddings after positional encoding
  Hidden-state index k → output of transformer layer k  (k = 1 … 24)

Students interact with the hidden states only through ``aggregation.py``.
``extract_hidden_states`` always returns **raw** per-layer tensors; call
``aggregation.aggregation_and_feature_extraction`` afterwards to obtain the
compact feature vectors used by the probe classifier.
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
_MAX_LENGTH = 512


def get_model_and_tokenizer(
    model_name: str = _DEFAULT_MODEL,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pre-trained Qwen2.5-0.5B model and its tokenizer.

    The model is loaded in ``bfloat16`` to keep the memory footprint small.
    Qwen2.5-0.5B fits comfortably on a free Google Colab T4 GPU (≈ 15 GB
    VRAM) and runs efficiently on CPU for smaller datasets.

    The model is returned in evaluation mode with hidden-state output enabled.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
                    ``"Qwen/Qwen2.5-0.5B"``.

    Returns:
        A ``(model, tokenizer)`` tuple.  The model is in eval mode.
    """
    print(f"[Model] Loading '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 4,
    max_length: int = _MAX_LENGTH,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract hidden states from all layers for a list of input texts.

    Processes texts in mini-batches and returns per-sample raw tensors.
    Call ``aggregation.aggregation_and_feature_extraction`` on the returned
    lists to produce compact feature vectors suitable for the probe classifier.

    Args:
        model:      Qwen2.5-0.5B model (from ``get_model_and_tokenizer``).
        tokenizer:  Corresponding tokenizer.
        texts:      List of input strings to encode.  Each string should be
                    the concatenation of the prompt and the generated response.
        device:     Device to run inference on.
        batch_size: Number of texts processed per forward pass.  Use a small
                    value (1–4) to stay within GPU memory limits on free Colab.
        max_length: Maximum token sequence length; inputs are truncated /
                    padded to this length.

    Returns:
        A ``(all_hidden_states, all_attention_masks)`` tuple where:

        - ``all_hidden_states[i]`` is a float32 tensor of shape
          ``(n_layers, seq_len, hidden_dim)`` for sample ``i``.
          ``n_layers = num_hidden_layers + 1`` (embedding layer + transformer
          layers).
        - ``all_attention_masks[i]`` is a 1-D tensor of length ``seq_len``
          with 1 for real tokens and 0 for padding.

    Note:
        Hidden states are returned on CPU to conserve GPU memory.
        The raw tensors can be large (25 × seq_len × 896 per sample);
        free them after aggregation to reclaim memory.
    """
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

        # outputs.hidden_states is a tuple of (num_hidden_layers + 1) tensors:
        # index 0 → token embeddings, indices 1 … num_hidden_layers → transformer layers.
        # Each tensor has shape (batch, seq_len, hidden_dim).
        hidden = torch.stack(outputs.hidden_states, dim=1)  # (batch, n_layers, seq_len, hidden_dim)
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            all_hidden_states.append(hidden[i].cpu().float())  # (n_layers, seq_len, hidden_dim)
            all_attention_masks.append(mask[i])                # (seq_len,)

    return all_hidden_states, all_attention_masks

