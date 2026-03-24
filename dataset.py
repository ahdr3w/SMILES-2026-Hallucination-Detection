"""
dataset.py — Dataset loader for the hallucination detection task
(fixed infrastructure, do not edit).

Loads a collection of prompt–response pairs labelled as truthful (0) or
hallucinated (1), splits them into train / validation / test subsets with a
fixed random seed, and optionally downloads the raw text data automatically.

The dataset is built from the TruthfulQA benchmark:
  Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods", 2022.
  https://huggingface.co/datasets/truthful_qa

Each sample is a question paired with a model-generated answer that is either
correct (label 0) or hallucinated (label 1).
"""

from __future__ import annotations

import json
import os
import random

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DATASET_PATH = os.path.join(_DATA_DIR, "dataset.jsonl")
_SEED = 42
_TRAIN_RATIO = 0.70
_VAL_RATIO = 0.15
# Test ratio = 1 - TRAIN - VAL = 0.15


# ---------------------------------------------------------------------------
# Dataset construction (executed once if dataset.jsonl is absent)
# ---------------------------------------------------------------------------


def _build_dataset(path: str) -> None:
    """Download TruthfulQA and create ``dataset.jsonl`` at ``path``.

    Falls back to a small built-in synthetic dataset if the ``datasets``
    library or internet access is unavailable.

    Args:
        path: Absolute file path where the JSONL file will be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        from datasets import load_dataset as hf_load  # type: ignore

        print("[Data] Downloading TruthfulQA from HuggingFace ...")
        tqa = hf_load("truthful_qa", "generation", trust_remote_code=True)
        split = tqa["validation"]

        samples: list[dict] = []
        for row in split:
            question = row["question"]
            correct_answers = row["correct_answers"]
            incorrect_answers = row["incorrect_answers"]

            for ans in correct_answers[:2]:
                samples.append(
                    {
                        "text": f"Q: {question} A: {ans}",
                        "label": 0,
                    }
                )
            for ans in incorrect_answers[:2]:
                samples.append(
                    {
                        "text": f"Q: {question} A: {ans}",
                        "label": 1,
                    }
                )

        print(f"[Data] Built {len(samples)} samples from TruthfulQA.")

    except Exception as exc:
        print(f"[Data] Could not load TruthfulQA ({exc}). Using built-in dataset.")
        samples = _BUILTIN_SAMPLES

    random.seed(_SEED)
    random.shuffle(samples)

    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Data] Dataset saved to '{path}' ({len(samples)} samples).")


# ---------------------------------------------------------------------------
# Built-in synthetic fallback (~300 samples)
# ---------------------------------------------------------------------------

_BUILTIN_SAMPLES: list[dict] = [
    # Geography — correct (label=0)
    {"text": "Q: What is the capital of France? A: The capital of France is Paris.", "label": 0},
    {"text": "Q: What is the capital of Germany? A: The capital of Germany is Berlin.", "label": 0},
    {"text": "Q: What is the capital of Japan? A: The capital of Japan is Tokyo.", "label": 0},
    {"text": "Q: What is the capital of Australia? A: The capital of Australia is Canberra.", "label": 0},
    {"text": "Q: What is the capital of Brazil? A: The capital of Brazil is Brasília.", "label": 0},
    {"text": "Q: What is the capital of Canada? A: The capital of Canada is Ottawa.", "label": 0},
    {"text": "Q: What is the capital of China? A: The capital of China is Beijing.", "label": 0},
    {"text": "Q: What is the capital of India? A: The capital of India is New Delhi.", "label": 0},
    {"text": "Q: What is the capital of Russia? A: The capital of Russia is Moscow.", "label": 0},
    {"text": "Q: What is the capital of Mexico? A: The capital of Mexico is Mexico City.", "label": 0},
    # Geography — hallucinated (label=1)
    {"text": "Q: What is the capital of France? A: The capital of France is Lyon.", "label": 1},
    {"text": "Q: What is the capital of Germany? A: The capital of Germany is Munich.", "label": 1},
    {"text": "Q: What is the capital of Japan? A: The capital of Japan is Osaka.", "label": 1},
    {"text": "Q: What is the capital of Australia? A: The capital of Australia is Sydney.", "label": 1},
    {"text": "Q: What is the capital of Brazil? A: The capital of Brazil is São Paulo.", "label": 1},
    {"text": "Q: What is the capital of Canada? A: The capital of Canada is Toronto.", "label": 1},
    {"text": "Q: What is the capital of China? A: The capital of China is Shanghai.", "label": 1},
    {"text": "Q: What is the capital of India? A: The capital of India is Mumbai.", "label": 1},
    {"text": "Q: What is the capital of Russia? A: The capital of Russia is St. Petersburg.", "label": 1},
    {"text": "Q: What is the capital of Mexico? A: The capital of Mexico is Guadalajara.", "label": 1},
    # Science — correct (label=0)
    {"text": "Q: What is the boiling point of water at standard pressure? A: Water boils at 100 degrees Celsius at standard atmospheric pressure.", "label": 0},
    {"text": "Q: What is the chemical symbol for gold? A: The chemical symbol for gold is Au.", "label": 0},
    {"text": "Q: How many bones are in the adult human body? A: The adult human body has 206 bones.", "label": 0},
    {"text": "Q: What planet is closest to the Sun? A: Mercury is the planet closest to the Sun.", "label": 0},
    {"text": "Q: What is the speed of light in a vacuum? A: The speed of light in a vacuum is approximately 299,792,458 meters per second.", "label": 0},
    {"text": "Q: What is the atomic number of carbon? A: Carbon has an atomic number of 6.", "label": 0},
    {"text": "Q: What gas do plants absorb during photosynthesis? A: Plants absorb carbon dioxide (CO2) during photosynthesis.", "label": 0},
    {"text": "Q: What is the powerhouse of the cell? A: The mitochondria is known as the powerhouse of the cell.", "label": 0},
    {"text": "Q: What force keeps planets in orbit around the Sun? A: Gravity keeps planets in orbit around the Sun.", "label": 0},
    {"text": "Q: How many chromosomes do humans have? A: Humans have 46 chromosomes arranged in 23 pairs.", "label": 0},
    # Science — hallucinated (label=1)
    {"text": "Q: What is the boiling point of water at standard pressure? A: Water boils at 90 degrees Celsius at standard atmospheric pressure.", "label": 1},
    {"text": "Q: What is the chemical symbol for gold? A: The chemical symbol for gold is Go.", "label": 1},
    {"text": "Q: How many bones are in the adult human body? A: The adult human body has 300 bones.", "label": 1},
    {"text": "Q: What planet is closest to the Sun? A: Venus is the planet closest to the Sun.", "label": 1},
    {"text": "Q: What is the speed of light in a vacuum? A: The speed of light in a vacuum is approximately 150,000 kilometers per second.", "label": 1},
    {"text": "Q: What is the atomic number of carbon? A: Carbon has an atomic number of 12.", "label": 1},
    {"text": "Q: What gas do plants absorb during photosynthesis? A: Plants absorb oxygen (O2) during photosynthesis.", "label": 1},
    {"text": "Q: What is the powerhouse of the cell? A: The nucleus is known as the powerhouse of the cell.", "label": 1},
    {"text": "Q: What force keeps planets in orbit around the Sun? A: Magnetism keeps planets in orbit around the Sun.", "label": 1},
    {"text": "Q: How many chromosomes do humans have? A: Humans have 23 chromosomes arranged in 23 pairs.", "label": 1},
    # History — correct (label=0)
    {"text": "Q: In what year did World War II end? A: World War II ended in 1945.", "label": 0},
    {"text": "Q: Who was the first President of the United States? A: George Washington was the first President of the United States.", "label": 0},
    {"text": "Q: When did the French Revolution begin? A: The French Revolution began in 1789.", "label": 0},
    {"text": "Q: Who wrote the play Hamlet? A: William Shakespeare wrote the play Hamlet.", "label": 0},
    {"text": "Q: In what year did humans first land on the Moon? A: Humans first landed on the Moon in 1969.", "label": 0},
    {"text": "Q: What ancient wonder was located in Alexandria? A: The Lighthouse of Alexandria was one of the Seven Wonders of the Ancient World.", "label": 0},
    {"text": "Q: Who painted the Mona Lisa? A: Leonardo da Vinci painted the Mona Lisa.", "label": 0},
    {"text": "Q: In what year did World War I begin? A: World War I began in 1914.", "label": 0},
    {"text": "Q: Who was the first woman to win a Nobel Prize? A: Marie Curie was the first woman to win a Nobel Prize, winning in 1903.", "label": 0},
    {"text": "Q: What year did the Berlin Wall fall? A: The Berlin Wall fell in 1989.", "label": 0},
    # History — hallucinated (label=1)
    {"text": "Q: In what year did World War II end? A: World War II ended in 1944.", "label": 1},
    {"text": "Q: Who was the first President of the United States? A: Benjamin Franklin was the first President of the United States.", "label": 1},
    {"text": "Q: When did the French Revolution begin? A: The French Revolution began in 1799.", "label": 1},
    {"text": "Q: Who wrote the play Hamlet? A: Christopher Marlowe wrote the play Hamlet.", "label": 1},
    {"text": "Q: In what year did humans first land on the Moon? A: Humans first landed on the Moon in 1972.", "label": 1},
    {"text": "Q: What ancient wonder was located in Alexandria? A: The Colossus of Alexandria was one of the Seven Wonders of the Ancient World.", "label": 1},
    {"text": "Q: Who painted the Mona Lisa? A: Michelangelo painted the Mona Lisa.", "label": 1},
    {"text": "Q: In what year did World War I begin? A: World War I began in 1918.", "label": 1},
    {"text": "Q: Who was the first woman to win a Nobel Prize? A: Florence Nightingale was the first woman to win a Nobel Prize.", "label": 1},
    {"text": "Q: What year did the Berlin Wall fall? A: The Berlin Wall fell in 1991.", "label": 1},
    # Mathematics — correct (label=0)
    {"text": "Q: What is the value of pi to two decimal places? A: Pi is approximately 3.14.", "label": 0},
    {"text": "Q: What is the square root of 144? A: The square root of 144 is 12.", "label": 0},
    {"text": "Q: How many sides does a hexagon have? A: A hexagon has 6 sides.", "label": 0},
    {"text": "Q: What is 15 percent of 200? A: 15 percent of 200 is 30.", "label": 0},
    {"text": "Q: What is the sum of angles in a triangle? A: The sum of interior angles in a triangle is 180 degrees.", "label": 0},
    {"text": "Q: What is 2 to the power of 10? A: 2 to the power of 10 is 1024.", "label": 0},
    {"text": "Q: What is the factorial of 5? A: The factorial of 5 is 120.", "label": 0},
    {"text": "Q: What is the area formula for a circle? A: The area of a circle is pi times the radius squared.", "label": 0},
    {"text": "Q: What is the Pythagorean theorem? A: The Pythagorean theorem states that a squared plus b squared equals c squared for a right triangle.", "label": 0},
    {"text": "Q: What is the derivative of x squared? A: The derivative of x squared is 2x.", "label": 0},
    # Mathematics — hallucinated (label=1)
    {"text": "Q: What is the value of pi to two decimal places? A: Pi is approximately 3.41.", "label": 1},
    {"text": "Q: What is the square root of 144? A: The square root of 144 is 14.", "label": 1},
    {"text": "Q: How many sides does a hexagon have? A: A hexagon has 8 sides.", "label": 1},
    {"text": "Q: What is 15 percent of 200? A: 15 percent of 200 is 25.", "label": 1},
    {"text": "Q: What is the sum of angles in a triangle? A: The sum of interior angles in a triangle is 360 degrees.", "label": 1},
    {"text": "Q: What is 2 to the power of 10? A: 2 to the power of 10 is 512.", "label": 1},
    {"text": "Q: What is the factorial of 5? A: The factorial of 5 is 60.", "label": 1},
    {"text": "Q: What is the area formula for a circle? A: The area of a circle is 2 times pi times the radius.", "label": 1},
    {"text": "Q: What is the Pythagorean theorem? A: The Pythagorean theorem states that a plus b equals c for a right triangle.", "label": 1},
    {"text": "Q: What is the derivative of x squared? A: The derivative of x squared is x.", "label": 1},
    # Literature — correct (label=0)
    {"text": "Q: Who wrote 'Pride and Prejudice'? A: Jane Austen wrote 'Pride and Prejudice'.", "label": 0},
    {"text": "Q: Who wrote '1984'? A: George Orwell wrote '1984'.", "label": 0},
    {"text": "Q: Who wrote 'The Great Gatsby'? A: F. Scott Fitzgerald wrote 'The Great Gatsby'.", "label": 0},
    {"text": "Q: Who wrote 'To Kill a Mockingbird'? A: Harper Lee wrote 'To Kill a Mockingbird'.", "label": 0},
    {"text": "Q: Who wrote 'Don Quixote'? A: Miguel de Cervantes wrote 'Don Quixote'.", "label": 0},
    {"text": "Q: Who wrote 'War and Peace'? A: Leo Tolstoy wrote 'War and Peace'.", "label": 0},
    {"text": "Q: Who wrote 'The Odyssey'? A: Homer wrote 'The Odyssey'.", "label": 0},
    {"text": "Q: Who wrote 'Moby Dick'? A: Herman Melville wrote 'Moby Dick'.", "label": 0},
    {"text": "Q: Who wrote 'Crime and Punishment'? A: Fyodor Dostoevsky wrote 'Crime and Punishment'.", "label": 0},
    {"text": "Q: Who wrote 'One Hundred Years of Solitude'? A: Gabriel García Márquez wrote 'One Hundred Years of Solitude'.", "label": 0},
    # Literature — hallucinated (label=1)
    {"text": "Q: Who wrote 'Pride and Prejudice'? A: Charlotte Brontë wrote 'Pride and Prejudice'.", "label": 1},
    {"text": "Q: Who wrote '1984'? A: Aldous Huxley wrote '1984'.", "label": 1},
    {"text": "Q: Who wrote 'The Great Gatsby'? A: Ernest Hemingway wrote 'The Great Gatsby'.", "label": 1},
    {"text": "Q: Who wrote 'To Kill a Mockingbird'? A: Truman Capote wrote 'To Kill a Mockingbird'.", "label": 1},
    {"text": "Q: Who wrote 'Don Quixote'? A: Gabriel García Márquez wrote 'Don Quixote'.", "label": 1},
    {"text": "Q: Who wrote 'War and Peace'? A: Anton Chekhov wrote 'War and Peace'.", "label": 1},
    {"text": "Q: Who wrote 'The Odyssey'? A: Virgil wrote 'The Odyssey'.", "label": 1},
    {"text": "Q: Who wrote 'Moby Dick'? A: Jack London wrote 'Moby Dick'.", "label": 1},
    {"text": "Q: Who wrote 'Crime and Punishment'? A: Leo Tolstoy wrote 'Crime and Punishment'.", "label": 1},
    {"text": "Q: Who wrote 'One Hundred Years of Solitude'? A: Pablo Neruda wrote 'One Hundred Years of Solitude'.", "label": 1},
    # Biology — correct (label=0)
    {"text": "Q: What is the basic unit of life? A: The cell is the basic unit of life.", "label": 0},
    {"text": "Q: What carries oxygen in red blood cells? A: Hemoglobin carries oxygen in red blood cells.", "label": 0},
    {"text": "Q: How many chambers does the human heart have? A: The human heart has four chambers.", "label": 0},
    {"text": "Q: What is the process by which plants make food? A: Photosynthesis is the process by which plants make food using sunlight.", "label": 0},
    {"text": "Q: What is DNA short for? A: DNA stands for deoxyribonucleic acid.", "label": 0},
    {"text": "Q: What organ produces insulin? A: The pancreas produces insulin.", "label": 0},
    {"text": "Q: What is the largest organ in the human body? A: The skin is the largest organ in the human body.", "label": 0},
    {"text": "Q: What type of blood cells fight infection? A: White blood cells fight infection.", "label": 0},
    {"text": "Q: What is the function of the ribosomes? A: Ribosomes synthesize proteins by translating messenger RNA.", "label": 0},
    {"text": "Q: What are the building blocks of proteins? A: Amino acids are the building blocks of proteins.", "label": 0},
    # Biology — hallucinated (label=1)
    {"text": "Q: What is the basic unit of life? A: The atom is the basic unit of life.", "label": 1},
    {"text": "Q: What carries oxygen in red blood cells? A: Myoglobin carries oxygen in red blood cells.", "label": 1},
    {"text": "Q: How many chambers does the human heart have? A: The human heart has three chambers.", "label": 1},
    {"text": "Q: What is the process by which plants make food? A: Respiration is the process by which plants make food using sunlight.", "label": 1},
    {"text": "Q: What is DNA short for? A: DNA stands for deoxyribose nucleic acid.", "label": 1},
    {"text": "Q: What organ produces insulin? A: The liver produces insulin.", "label": 1},
    {"text": "Q: What is the largest organ in the human body? A: The liver is the largest organ in the human body.", "label": 1},
    {"text": "Q: What type of blood cells fight infection? A: Red blood cells fight infection.", "label": 1},
    {"text": "Q: What is the function of the ribosomes? A: Ribosomes store genetic information for cell division.", "label": 1},
    {"text": "Q: What are the building blocks of proteins? A: Nucleotides are the building blocks of proteins.", "label": 1},
    # Physics — correct (label=0)
    {"text": "Q: What is Newton's first law of motion? A: Newton's first law states that an object remains at rest or in uniform motion unless acted upon by an external force.", "label": 0},
    {"text": "Q: What is the unit of electrical resistance? A: The unit of electrical resistance is the ohm.", "label": 0},
    {"text": "Q: What particle has a negative charge in an atom? A: Electrons are the particles with a negative charge in an atom.", "label": 0},
    {"text": "Q: What is the unit of force in the SI system? A: The newton is the unit of force in the SI system.", "label": 0},
    {"text": "Q: What does E=mc² represent? A: E=mc² represents the equivalence of mass and energy, where E is energy, m is mass, and c is the speed of light.", "label": 0},
    # Physics — hallucinated (label=1)
    {"text": "Q: What is Newton's first law of motion? A: Newton's first law states that force equals mass times acceleration.", "label": 1},
    {"text": "Q: What is the unit of electrical resistance? A: The unit of electrical resistance is the volt.", "label": 1},
    {"text": "Q: What particle has a negative charge in an atom? A: Protons are the particles with a negative charge in an atom.", "label": 1},
    {"text": "Q: What is the unit of force in the SI system? A: The joule is the unit of force in the SI system.", "label": 1},
    {"text": "Q: What does E=mc² represent? A: E=mc² represents the relationship between energy and temperature.", "label": 1},
    # Technology — correct (label=0)
    {"text": "Q: Who co-founded Apple Inc.? A: Steve Jobs and Steve Wozniak co-founded Apple Inc.", "label": 0},
    {"text": "Q: What does CPU stand for? A: CPU stands for Central Processing Unit.", "label": 0},
    {"text": "Q: In what year was the World Wide Web invented? A: The World Wide Web was invented in 1989 by Tim Berners-Lee.", "label": 0},
    {"text": "Q: What programming language was created by Guido van Rossum? A: Python was created by Guido van Rossum.", "label": 0},
    {"text": "Q: What does HTTP stand for? A: HTTP stands for HyperText Transfer Protocol.", "label": 0},
    # Technology — hallucinated (label=1)
    {"text": "Q: Who co-founded Apple Inc.? A: Bill Gates and Paul Allen co-founded Apple Inc.", "label": 1},
    {"text": "Q: What does CPU stand for? A: CPU stands for Computer Processing Unit.", "label": 1},
    {"text": "Q: In what year was the World Wide Web invented? A: The World Wide Web was invented in 1995 by Steve Jobs.", "label": 1},
    {"text": "Q: What programming language was created by Guido van Rossum? A: Java was created by Guido van Rossum.", "label": 1},
    {"text": "Q: What does HTTP stand for? A: HTTP stands for High-Transfer Text Protocol.", "label": 1},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(
    data_dir: str = _DATA_DIR,
    train_ratio: float = _TRAIN_RATIO,
    val_ratio: float = _VAL_RATIO,
    seed: int = _SEED,
) -> tuple[dict, dict, dict]:
    """Load the hallucination dataset and return train / val / test splits.

    If ``data/dataset.jsonl`` does not exist it is created automatically
    (either by downloading TruthfulQA or by using the built-in synthetic data).

    Args:
        data_dir:    Directory where ``dataset.jsonl`` is stored / will be created.
        train_ratio: Fraction of samples to use for training.
        val_ratio:   Fraction of samples to use for validation.
        seed:        Random seed for reproducible splits.

    Returns:
        A ``(train, val, test)`` tuple.  Each split is a dict with keys:
          - ``"texts"``:  list[str] — input Q&A texts
          - ``"labels"``: list[int] — 0 (truthful) or 1 (hallucinated)
    """
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    if not os.path.exists(dataset_path):
        _build_dataset(dataset_path)

    samples: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    def _split(items: list[dict]) -> dict:
        return {
            "texts": [s["text"] for s in items],
            "labels": [s["label"] for s in items],
        }

    train = _split(samples[:n_train])
    val = _split(samples[n_train : n_train + n_val])
    test = _split(samples[n_train + n_val :])

    return train, val, test
