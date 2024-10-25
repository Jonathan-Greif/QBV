# Query-by-Vocal Imitation (QBV)

In this repository, we publish the pre-trained models and the code described in the paper:

- _Improving Query-by-Vocal Imitation with Contrastive Learning and Audio Pretraining_. The paper has been presented in DCASE 2024.

## Abstract

Query-by-Vocal Imitation (QBV) is about searching audio files within databases using vocal imitations created by the user’s voice.
Since most humans can effectively communicate sound concepts through voice, QBV offers the more intuitive and convenient approach compared to text-based search. 
To fully leverage QBV, developing robust audio feature representations for both the vocal imitation and the original sound is crucial. 
In this paper, we present a new system for QBV that utilizes the feature extraction capabilities of Convolutional Neural Networks pre-trained with large-scale general-purpose audio datasets. 
We integrate these pre-trained models into a dual encoder architecture and fine-tune them end-to-end using contrastive learning. 
A distinctive aspect of our proposed method is the fine-tuning strategy of pre-trained models using an adapted NT-Xent loss for contrastive learning, creating a shared embedding space for reference recordings and vocal imitations. 
The proposed system significantly enhances audio retrieval performance, establishing a new state of the art on both coarse- and fine-grained QBV tasks.

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
