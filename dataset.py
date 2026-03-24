"""
dataset.py — Dataset loader for the hallucination detection task
(fixed infrastructure, do not edit).

Loads a CSV dataset of prompt–response pairs labelled as truthful (0) or
hallucinated (1), splits them into train / validation / test subsets, and
prepares text inputs for hidden-state extraction.

Expected CSV columns
--------------------
  id, instruction, question, prompt_template, gold_response,
  generated_response, response_history, context, prompt, hallucination

The ``prompt`` column (or ``prompt_template`` + ``question`` columns) provides
the input given to the LLM; ``generated_response`` is the model's output;
``hallucination`` is the ground-truth binary label (0 = truthful, 1 = hallucinated).

Shadowing of ``gold_response``
------------------------------
The test split is the held-out evaluation set.  To prevent students from using
the gold answer to build a trivial detector (e.g., exact-match comparison),
``gold_response`` is set to ``None`` in the test split before it is returned.
"""

from __future__ import annotations

import ast
import os
from typing import Union

import pandas as pd

from splitting import split_data

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DEFAULT_CSV = "dataset.csv"

columns = [
    "id",
    "instruction",
    "question",
    "prompt_template",
    "gold_response",
    "generated_response",
    "response_history",
    "context",
    "prompt",
    "hallucination",
]


# ---------------------------------------------------------------------------
# CSV loader (provided by competition organisers — do not edit)
# ---------------------------------------------------------------------------


def load_dataset(
    file_path: str,
    test_size: Union[int, float] = 0.2,
    random_state: int = 0,
) -> pd.DataFrame:
    """Load a dataset from a CSV file and perform basic preprocessing.

    Constructs the ``prompt`` column from ``prompt_template`` when the column
    is absent.  Parses list-valued cells stored as Python literals.  Fills any
    missing optional columns with ``None``.

    Args:
        file_path:    Path to the CSV file.
        test_size:    Reserved for caller-side splitting (not used here).
        random_state: Reserved for caller-side splitting (not used here).

    Returns:
        A pandas DataFrame with exactly the columns listed in ``columns``.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If neither ``prompt`` nor ``prompt_template`` columns are
                    present.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {file_path}: {e}")

    df = df.dropna(axis=1, how="all")
    missing = [col for col in columns if col not in df.columns]

    if "prompt" in missing:
        if "prompt_template" in missing:
            raise ValueError(
                "If DataFrame does not contain `prompt` column then "
                "prompt_template must be provided."
            )
        else:
            def build_prompt(row: pd.Series):
                question = row["question"] if "question" in row.index else None
                context = row["context"] if "context" in row.index else None
                instruction = row["instruction"] if "instruction" in row.index else None
                prompt_template = row["prompt_template"]
                if question is not None:
                    try:
                        question = ast.literal_eval(question)
                    except Exception:
                        pass
                questions_series = None
                if isinstance(question, list):
                    questions_series, question = question, question[0]
                fields: dict = {}
                fields["question"] = question
                if context is not None:
                    fields["context"] = context
                if instruction is not None:
                    fields["instruction"] = instruction
                prompt = prompt_template.format(**fields)
                if questions_series:
                    return [prompt] + questions_series[1:]
                return prompt

            df["prompt"] = [build_prompt(row) for _, row in df.iterrows()]
            missing.remove("prompt")
    else:
        try:
            df["prompt"] = df["prompt"].apply(lambda x: ast.literal_eval(x))
        except Exception:
            pass

    if "response_history" not in missing:
        df["response_history"] = df["response_history"].apply(
            lambda x: ast.literal_eval(x)
        )
    if "generated_response" not in missing:
        df["generated_response"] = df["generated_response"].apply(
            lambda x: ast.literal_eval(x)
            if isinstance(x, str) and x.startswith("[") and x.endswith("]")
            else x
        )
    if "id" in missing:
        df["id"] = df.index.values
        missing.remove("id")
    if "hallucination" not in missing:
        df = df[~df["hallucination"].isna()]

    for col in missing:
        df[col] = None

    return df[columns]


# ---------------------------------------------------------------------------
# Text construction helpers
# ---------------------------------------------------------------------------


def build_text(row: pd.Series) -> str:
    """Construct the text to encode from a single dataset row.

    Concatenates the prompt and the generated response so that the LLM's
    hidden states capture both the question context and the (potentially
    hallucinated) answer.

    Args:
        row: A pandas Series corresponding to one dataset row.

    Returns:
        A single string: ``"{prompt}\\n{generated_response}"``.
    """
    prompt = row["prompt"]
    if isinstance(prompt, list):
        prompt = prompt[0]
    prompt = str(prompt) if prompt is not None else ""

    response = row["generated_response"]
    if isinstance(response, list):
        response = response[0] if response else ""
    response = str(response) if response is not None else ""

    return f"{prompt}\n{response}"


# Keep the private alias for internal use.
_build_text = build_text


# ---------------------------------------------------------------------------
# Fallback synthetic dataset
# ---------------------------------------------------------------------------


def _build_fallback_csv(path: str) -> None:
    """Create a small synthetic CSV at ``path`` matching the expected schema.

    Used when no real dataset file is available (e.g., offline environments).
    Contains balanced factual Q&A pairs across several knowledge domains.

    Args:
        path: Absolute file path where the CSV will be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _TEMPLATE = "Answer the following question.\nQuestion: {question}"

    _QA: list[tuple[str, str, str, int]] = [
        # (question, gold_response, generated_response, hallucination)
        # Geography
        ("What is the capital of France?", "Paris", "The capital of France is Paris.", 0),
        ("What is the capital of Germany?", "Berlin", "The capital of Germany is Berlin.", 0),
        ("What is the capital of Japan?", "Tokyo", "The capital of Japan is Tokyo.", 0),
        ("What is the capital of Australia?", "Canberra", "The capital of Australia is Canberra.", 0),
        ("What is the capital of Brazil?", "Brasília", "The capital of Brazil is Brasília.", 0),
        ("What is the capital of France?", "Paris", "The capital of France is Lyon.", 1),
        ("What is the capital of Germany?", "Berlin", "The capital of Germany is Munich.", 1),
        ("What is the capital of Japan?", "Tokyo", "The capital of Japan is Osaka.", 1),
        ("What is the capital of Australia?", "Canberra", "The capital of Australia is Sydney.", 1),
        ("What is the capital of Brazil?", "Brasília", "The capital of Brazil is São Paulo.", 1),
        # Science
        ("What is the chemical symbol for gold?", "Au", "The chemical symbol for gold is Au.", 0),
        ("How many bones are in the adult human body?", "206", "The adult human body has 206 bones.", 0),
        ("What planet is closest to the Sun?", "Mercury", "Mercury is the planet closest to the Sun.", 0),
        ("What is the atomic number of carbon?", "6", "Carbon has an atomic number of 6.", 0),
        ("What is the powerhouse of the cell?", "Mitochondria", "The mitochondria is the powerhouse of the cell.", 0),
        ("What is the chemical symbol for gold?", "Au", "The chemical symbol for gold is Go.", 1),
        ("How many bones are in the adult human body?", "206", "The adult human body has 300 bones.", 1),
        ("What planet is closest to the Sun?", "Mercury", "Venus is the planet closest to the Sun.", 1),
        ("What is the atomic number of carbon?", "6", "Carbon has an atomic number of 12.", 1),
        ("What is the powerhouse of the cell?", "Mitochondria", "The nucleus is the powerhouse of the cell.", 1),
        # History
        ("In what year did World War II end?", "1945", "World War II ended in 1945.", 0),
        ("Who was the first US President?", "George Washington", "George Washington was the first US President.", 0),
        ("Who wrote the play Hamlet?", "William Shakespeare", "William Shakespeare wrote Hamlet.", 0),
        ("In what year did humans first land on the Moon?", "1969", "Humans first landed on the Moon in 1969.", 0),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Leonardo da Vinci painted the Mona Lisa.", 0),
        ("In what year did World War II end?", "1945", "World War II ended in 1944.", 1),
        ("Who was the first US President?", "George Washington", "Benjamin Franklin was the first US President.", 1),
        ("Who wrote the play Hamlet?", "William Shakespeare", "Christopher Marlowe wrote Hamlet.", 1),
        ("In what year did humans first land on the Moon?", "1969", "Humans first landed on the Moon in 1972.", 1),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Michelangelo painted the Mona Lisa.", 1),
        # Mathematics
        ("What is the square root of 144?", "12", "The square root of 144 is 12.", 0),
        ("How many sides does a hexagon have?", "6", "A hexagon has 6 sides.", 0),
        ("What is the sum of angles in a triangle?", "180 degrees", "The sum of angles in a triangle is 180 degrees.", 0),
        ("What is 2 to the power of 10?", "1024", "2 to the power of 10 is 1024.", 0),
        ("What is the factorial of 5?", "120", "The factorial of 5 is 120.", 0),
        ("What is the square root of 144?", "12", "The square root of 144 is 14.", 1),
        ("How many sides does a hexagon have?", "6", "A hexagon has 8 sides.", 1),
        ("What is the sum of angles in a triangle?", "180 degrees", "The sum of angles in a triangle is 360 degrees.", 1),
        ("What is 2 to the power of 10?", "1024", "2 to the power of 10 is 512.", 1),
        ("What is the factorial of 5?", "120", "The factorial of 5 is 60.", 1),
        # Literature
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen", "Jane Austen wrote 'Pride and Prejudice'.", 0),
        ("Who wrote '1984'?", "George Orwell", "George Orwell wrote '1984'.", 0),
        ("Who wrote 'The Great Gatsby'?", "F. Scott Fitzgerald", "F. Scott Fitzgerald wrote 'The Great Gatsby'.", 0),
        ("Who wrote 'Don Quixote'?", "Miguel de Cervantes", "Miguel de Cervantes wrote 'Don Quixote'.", 0),
        ("Who wrote 'War and Peace'?", "Leo Tolstoy", "Leo Tolstoy wrote 'War and Peace'.", 0),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen", "Charlotte Brontë wrote 'Pride and Prejudice'.", 1),
        ("Who wrote '1984'?", "George Orwell", "Aldous Huxley wrote '1984'.", 1),
        ("Who wrote 'The Great Gatsby'?", "F. Scott Fitzgerald", "Ernest Hemingway wrote 'The Great Gatsby'.", 1),
        ("Who wrote 'Don Quixote'?", "Miguel de Cervantes", "Gabriel García Márquez wrote 'Don Quixote'.", 1),
        ("Who wrote 'War and Peace'?", "Leo Tolstoy", "Anton Chekhov wrote 'War and Peace'.", 1),
        # Biology
        ("What is the basic unit of life?", "Cell", "The cell is the basic unit of life.", 0),
        ("What carries oxygen in red blood cells?", "Hemoglobin", "Hemoglobin carries oxygen in red blood cells.", 0),
        ("How many chambers does the human heart have?", "4", "The human heart has four chambers.", 0),
        ("What is DNA short for?", "Deoxyribonucleic acid", "DNA stands for deoxyribonucleic acid.", 0),
        ("What organ produces insulin?", "Pancreas", "The pancreas produces insulin.", 0),
        ("What is the basic unit of life?", "Cell", "The atom is the basic unit of life.", 1),
        ("What carries oxygen in red blood cells?", "Hemoglobin", "Myoglobin carries oxygen in red blood cells.", 1),
        ("How many chambers does the human heart have?", "4", "The human heart has three chambers.", 1),
        ("What is DNA short for?", "Deoxyribonucleic acid", "DNA stands for deoxyribose nucleic acid.", 1),
        ("What organ produces insulin?", "Pancreas", "The liver produces insulin.", 1),
        # Technology
        ("Who co-founded Apple Inc.?", "Steve Jobs and Steve Wozniak", "Steve Jobs and Steve Wozniak co-founded Apple Inc.", 0),
        ("What does CPU stand for?", "Central Processing Unit", "CPU stands for Central Processing Unit.", 0),
        ("What programming language did Guido van Rossum create?", "Python", "Python was created by Guido van Rossum.", 0),
        ("What does HTTP stand for?", "HyperText Transfer Protocol", "HTTP stands for HyperText Transfer Protocol.", 0),
        ("In what year was the World Wide Web invented?", "1989", "The World Wide Web was invented in 1989 by Tim Berners-Lee.", 0),
        ("Who co-founded Apple Inc.?", "Steve Jobs and Steve Wozniak", "Bill Gates and Paul Allen co-founded Apple Inc.", 1),
        ("What does CPU stand for?", "Central Processing Unit", "CPU stands for Computer Processing Unit.", 1),
        ("What programming language did Guido van Rossum create?", "Python", "Java was created by Guido van Rossum.", 1),
        ("What does HTTP stand for?", "HyperText Transfer Protocol", "HTTP stands for High-Transfer Text Protocol.", 1),
        ("In what year was the World Wide Web invented?", "1989", "The World Wide Web was invented in 1995 by Steve Jobs.", 1),
    ]

    rows = []
    for i, (question, gold, generated, hallucination) in enumerate(_QA):
        rows.append(
            {
                "id": i,
                "instruction": None,
                "question": question,
                "prompt_template": _TEMPLATE,
                "gold_response": gold,
                "generated_response": generated,
                "response_history": None,
                "context": None,
                "prompt": _TEMPLATE.format(question=question),
                "hallucination": hallucination,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[Data] Fallback dataset saved to '{path}' ({len(rows)} samples).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(
    file_path: str | None = None,
    data_dir: str = _DATA_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[dict, dict, dict]:
    """Load the hallucination dataset and return train / val / test splits.

    Locates the CSV at ``file_path`` (if given) or ``data_dir/dataset.csv``.
    If neither exists, a built-in synthetic fallback CSV is created at
    ``data_dir/dataset.csv``.

    The ``gold_response`` column is set to ``None`` in the returned **test**
    split so that students cannot use the gold answer to build a trivial
    detector.

    Args:
        file_path:    Explicit path to the dataset CSV.  When ``None``,
                      ``data_dir/dataset.csv`` is used.
        data_dir:     Directory where ``dataset.csv`` is stored / will be
                      created when ``file_path`` is ``None``.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A ``(train, val, test)`` tuple.  Each split is a dict with keys:

        - ``"texts"``:  list[str] — combined prompt + generated_response text
        - ``"labels"``: list[int] — 0 (truthful) or 1 (hallucinated)

    Note:
        ``gold_response`` is ``None`` for every row in the test split.
    """
    if file_path is None:
        file_path = os.path.join(data_dir, _DEFAULT_CSV)

    if not os.path.exists(file_path):
        print(f"[Data] '{file_path}' not found. Creating fallback dataset ...")
        _build_fallback_csv(file_path)

    df = load_dataset(file_path, random_state=random_state)

    # Delegate splitting to the student-implementable splitting module.
    train_df, val_df, test_df = split_data(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # Shadow gold_response in the test split
    test_df = test_df.copy()
    test_df["gold_response"] = None

    def _df_to_split(df_split: pd.DataFrame) -> dict:
        return {
            "texts": [_build_text(row) for _, row in df_split.iterrows()],
            "labels": [int(bool(h)) for h in df_split["hallucination"]],
        }

    return _df_to_split(train_df), _df_to_split(val_df), _df_to_split(test_df)


def load_dataframe(
    file_path: str | None = None,
    data_dir: str = _DATA_DIR,
) -> pd.DataFrame:
    """Load the hallucination dataset and return the full preprocessed DataFrame.

    Unlike ``load_data``, this function does **not** split the data — the caller
    is responsible for splitting.  This is the entry point used by the
    evaluation notebook so that students can define their own splitting strategy
    (random splits, k-fold, group-aware, etc.) directly in the notebook.

    Creates the fallback synthetic CSV if the dataset file is absent.

    Args:
        file_path: Explicit path to the dataset CSV.  When ``None``,
                   ``data_dir/dataset.csv`` is used.
        data_dir:  Directory where ``dataset.csv`` is stored / will be created
                   when ``file_path`` is ``None``.

    Returns:
        A pandas DataFrame with exactly the columns listed in ``columns``.
    """
    if file_path is None:
        file_path = os.path.join(data_dir, _DEFAULT_CSV)

    if not os.path.exists(file_path):
        print(f"[Data] '{file_path}' not found. Creating fallback dataset ...")
        _build_fallback_csv(file_path)

    return load_dataset(file_path)
