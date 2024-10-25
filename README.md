# Query-by-Vocal Imitation (QBV)


## Getting Started

```
conda create -n qbv python=3.8

conda activate qbv

pip install -r requirements.txt
```

## Experiments 

### Training

Coarse-grained
```
python ex_qbv.py --roll --fold=0 --id=001
```
Fine-grained
```
python ex_qbv.py --roll --fine_grained --id=001
```

### Testing

Coarse-grained
```
python test_coarse.py --own_module

python test_coarse.py --arch=M-VGGish --sr_down=16000 --dur=15.4

python test_coarse.py --arch=2DFT --sr_down=8000 --dur=15.4
```
Fine-grained
```
python test_fine.py --own_module

python test_fine.py --arch=M-VGGish --sr_down=16000 --dur=15.4

python test_fine.py --arch=2DFT --sr_down=8000 --dur=15.4
```
