# Query-by-Vocal Imitation (QBV)

The code for testing the dual encoders will appear soon.
Also weights will be released.

## Getting Started

```
conda create -n qbv python=3.8

conda activate qbv

pip install -r requirements.txt
```

## Experiments 

Coarse-grained
```
python ex_qbv.py --roll --fold=0 --id=001
```
Fine-grained
```
python ex_qbv.py --roll --fine_grained --id=001
```
