# edians

Extract educational instruments from PDF files. This is an NER task. The code in this repository allows you to train an NER model and evaluate it on new data.

## Activating Conda Environment
```bash
conda create -f environment.yml
conda activate edians
```

## Data

You will need to populate your `data/` directory with the following structure:
```
data
├── labels
│   └── labels.csv
└── pdfs
    ├── 001.pdf
    ├── 002.pdf
    (...)
```
## Training

```bash
python src/main.py --labels data/labels/labels.csv --data data/pdfs --log debug
```