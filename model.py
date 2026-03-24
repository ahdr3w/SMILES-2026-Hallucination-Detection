"""
model.py — LLM loader (fixed infrastructure, do not edit).

Loads ``Qwen/Qwen2.5-0.5B`` and exposes a single helper,
``get_model_and_tokenizer``, that returns the model and tokenizer ready for
inference.

The hidden-state extraction loop and the aggregation step are written
**explicitly in the notebook** (Section 4) so you can see — and learn from —
every step of the pipeline.  The key constants used by that loop are:

  ``_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"``
  ``MAX_LENGTH = 512``  (also set in the notebook config cell)

Qwen2.5-0.5B architecture reference (``model.config``)
--------------------------------------------------------
  ``model.config.num_hidden_layers``  — number of transformer layers (24)
  ``model.config.hidden_size``        — hidden dimension per token  (896)
  Hidden-state index 0 → token embeddings after positional encoding
  Hidden-state index k → output of transformer layer k  (k = 1 … 24)

The model is a **decoder-only** (causal) language model, so the last real
(non-padding) token attends to the entire preceding context and is the
natural choice for sequence-level pooling.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512


def get_model_and_tokenizer(
    model_name: str = _DEFAULT_MODEL,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pre-trained Qwen2.5-0.5B model and its tokenizer.

    The model is loaded in ``bfloat16`` to keep the memory footprint small.
    Qwen2.5-0.5B fits comfortably on a free Google Colab T4 GPU (≈ 15 GB
    VRAM) and runs efficiently on CPU for smaller datasets.

    The model is returned in evaluation mode with hidden-state output enabled
    (``output_hidden_states=True``), so every forward pass returns a tuple of
    per-layer activation tensors.

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

