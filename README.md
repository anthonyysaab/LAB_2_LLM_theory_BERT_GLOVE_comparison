# Lab 2 - BERT and GloVe Comparison

This repository contains Lab 2 for the LLM theory course at Paris Cite. The project compares distributional properties of BERT-style contextual embeddings and GloVe static word embeddings through similarity, morphology, anisotropy, and neighborhood analyses.

## Contents

- `lab2_pipeline.py` - main experiment pipeline.
- `q8_neighborhood_glove.py` - additional GloVe neighborhood analysis.
- `morph_families.tsv` - morphology family input table.
- `Outputs/` - generated plots, tables, and report support files.
- `lab2.pdf` - submitted lab report.
- `glove.6B.100d` - note file explaining the external GloVe download requirement.

## Setup

```bash
pip install -r requirements.txt
```

## External Data

The full GloVe vector file is not stored in this repository. Download `glove.6B.100d.txt` from the Stanford GloVe release page and place it where the scripts expect it before running the full pipeline.

The pipeline may also create local cache files under `data/cache/`; those are ignored by Git.

## Run

```bash
python lab2_pipeline.py
python q8_neighborhood_glove.py
```

## Notes

Generated figures and tables from the submitted run are kept in `Outputs/`. Large local downloads and cache files should stay outside Git.
