"""
model.py — LLM loader (fixed infrastructure, do not edit).

Loads ``Qwen/Qwen2.5-0.5B`` and exposes ``get_model_and_tokenizer``, which
returns the model and tokenizer ready for inference with hidden-state output
enabled (``output_hidden_states=True``).

Key constants used by the hidden-state extraction loop in ``solution.ipynb``:

  ``_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"``
  ``MAX_LENGTH = 512``
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512


def get_model_and_tokenizer(
    model_name: str = _DEFAULT_MODEL,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pre-trained model and its tokenizer.

    The model is loaded in ``bfloat16``, set to eval mode, and configured to
    return hidden states on every forward pass.

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

