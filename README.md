# AI Duplicate Bug Finder

## Overview
A Python prototype that identifies potentially duplicate game bug reports using semantic similarity.

## How it works
1. Loads sample bug reports from a CSV file.
2. Converts bug descriptions into embeddings using Sentence Transformers.
3. Calculates cosine similarity between bug descriptions.
4. Flags pairs whose similarity exceeds a configurable threshold.
5. Evaluates predictions against labelled test pairs using accuracy, precision, and recall.

## Technologies
- Python
- Sentence Transformers
- Embeddings and cosine similarity
- CSV data processing
- pytest

## Project files
- `main.py` — reads and displays bug reports.
- `duplicate_finder.py` — identifies potentially similar bug reports.
- `evaluate.py` — evaluates predictions against labelled examples.
- `test_evaluate.py` — automated tests for threshold logic.
- `data/bugs.csv` — sample, fictional bug reports.

## Running the project
Run these commands from the project directory:

```bash
python duplicate_finder.py
python evaluate.py
python -m pytest -v
```

## Limitations
The similarity threshold is experimental. The evaluation dataset is small and uses manually labelled pairs, so results do not establish real-world accuracy. Larger, independent test data is needed.

## Purpose
Built as a personal learning project exploring AI-assisted QA workflows and semantic duplicate detection.