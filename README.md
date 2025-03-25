

# 🚨 Critical Issue in the Dataloader 🚨  

If you cloned this repository **before 25.03.2025**, there was an small but critical **issue in the dataloader** that basically hindered training. Sorry for the inconvenience!


# Query-by-Vocal Imitation (QBV)

In this repository, we publish the model checkpoints and the code described in the paper:

- **Title**: [Improving Query-by-Vocal Imitation with Contrastive Learning and Audio Pretraining](https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Greif_36.pdf)
- **Authors**: Jonathan Greif, Florian Schmid, Paul Primus, Gerhard Widmer
- **Workshop**: Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events, (DCASE), Tokyo, Japan, 2024
- **Award**: Best Student Paper Award

## Abstract

Query-by-Vocal Imitation (QBV) is about searching audio files within databases using vocal imitations created by the user’s voice.
Since most humans can effectively communicate sound concepts through voice, QBV offers the more intuitive and convenient approach compared to text-based search. 
To fully leverage QBV, developing robust audio feature representations for both the vocal imitation and the original sound is crucial. 
In this paper, we present a new system for QBV that utilizes the feature extraction capabilities of Convolutional Neural Networks pre-trained with large-scale general-purpose audio datasets. 
We integrate these pre-trained models into a dual encoder architecture and fine-tune them end-to-end using contrastive learning. 
A distinctive aspect of our proposed method is the fine-tuning strategy of pre-trained models using an adapted NT-Xent loss for contrastive learning, creating a shared embedding space for reference recordings and vocal imitations. 
The proposed system significantly enhances audio retrieval performance, establishing a new state of the art on both coarse- and fine-grained QBV tasks.


## Getting Started

First, create a new Conda environment and install the required dependencies:  

```
conda create -n qbv python=3.8

conda activate qbv

pip install -r requirements.txt
```

## Experiments 
The settings are detailed in Section 4 of the paper.

### Training

Train the model in the coarse-grained QBV setting:
```
python ex_qbv.py --roll --fold=0 --id=001 --save_model
```
Train the model in the fine-grained QBV setting:
```
python ex_qbv.py --roll --fine_grained --id=001 --save_model
```

### Testing

#### Coarse-grained Testing

Run the default test:
```
python test_coarse.py --own_module
```
Test with M-VGGish architecture, 16kHz sampling rate, and item duration of 15.4s:
```
python test_coarse.py --arch=M-VGGish --sr_down=16000 --dur=15.4
```
Test with 2DFT architecture, 8kHz sampling rate, and item duration of 15.4s:
```
python test_coarse.py --arch=2DFT --sr_down=8000 --dur=15.4
```
#### Fine-grained Testing

Run the default fine-grained test:
```
python test_fine.py --own_module
```
Test with M-VGGish architecture, 16kHz sampling rate, and item duration of 15.4s:
```
python test_fine.py --arch=M-VGGish --sr_down=16000 --dur=15.4
```
Test with 2DFT architecture, 8kHz sampling rate, and item duration of 15.4s:
```
python test_fine.py --arch=2DFT --sr_down=8000 --dur=15.4
```

## Contact

For questions or inquiries, please don't hesitate to contact me at [jonathan.greif@jku.at](mailto:jonathan.greif@jku.at).

## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{Greif2024,
    author = "Greif, Jonathan and Schmid, Florian and Primus, Paul and Widmer, Gerhard",
    title = "Improving Query-By-Vocal Imitation with Contrastive Learning and Audio Pretraining",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024)",
    address = "Tokyo, Japan",
    month = "October",
    year = "2024",
    pages = "51--55"
}
```
