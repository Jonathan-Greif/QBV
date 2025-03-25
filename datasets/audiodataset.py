import os
import numpy as np
import pandas as pd
import librosa
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from typing import Callable, List

# Adapted from: https://github.com/fschmid56/malach23-pipeline/blob/main/datasets/audiodataset.py

dataset_dir, dataset_config = "", {}


def set_directory(fine_grained=False, fold=0):
    global dataset_dir
    global dataset_config
    if fine_grained:
        dataset_dir = os.path.join("datasets", "data", "fine grained", "VocalSketch")
        dataset_config = {
            "meta_csv": os.path.join(dataset_dir, "meta.csv"),
            "train_files_csv": os.path.join(dataset_dir, "train.csv"),
            "val_files_csv": os.path.join(dataset_dir, "val.csv")
        }
    else:
        dataset_dir = os.path.join("datasets", "data", "coarse grained")
        dataset_config = {
            "meta_csv": os.path.join(dataset_dir, "splits", "meta.csv"),
            "train_files_csv": os.path.join(dataset_dir, "splits", f"fold{fold}", "train.csv"),
            "val_files_csv": os.path.join(dataset_dir, "splits", f"fold{fold}", "val.csv")
        }


class BasicAudioDataset(Dataset):
    def __init__(self, meta_csv: str, sr: int = 32000, duration: float = 10.0,
                 gain_augment: int = 0, padding: str = "zero", cache_path: str = None):
        """
        @param meta_csv: meta csv file for the dataset
        @param sr: sampling rate
        @param duration: how much of the audio file is used
        @param gain_augment: modifies the amplitude of raw samples in the time domain
        @param padding: is padding used and how?
        @param cache_path: cache path to store resampled waveforms
        return: vocal imitation (waveform); sound recording (waveform);
                0 or 1 (indicating whether it's a negative or a positive pair); label of vocal imitation
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()  # sklearn label encoder to transform strings into numbers
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1))).long()
        self.files = df[['filename']].values.reshape(-1)
        self.positive_pairs = df[['positive_pair']].values.reshape(-1)
        self.negative_pairs = df[['negative_pair']].values.reshape(-1)
        self.sr = sr
        self.dur = duration
        self.gain_augment = gain_augment
        self.padding = padding
        if cache_path is not None:
            self.cache_path_vocal = os.path.join(dataset_dir, cache_path,
                                                 f"vocal_r{self.sr}_d{int(self.dur)}_{self.padding}padding",
                                                 "files_cache")
            os.makedirs(self.cache_path_vocal, exist_ok=True)

            self.cache_path_rec = os.path.join(dataset_dir, cache_path,
                                               f"rec_r{self.sr}_d{int(self.dur)}_{self.padding}padding",
                                               "files_cache")
            os.makedirs(self.cache_path_rec, exist_ok=True)
        else:
            self.cache_path_vocal = None
            self.cache_path_rec = None

    def __getitem__(self, index):
        index, pos_or_neg = index
        if self.cache_path_vocal:
            cpath_vocal = os.path.join(self.cache_path_vocal, str(index) + ".pt")
            imitation = self._getitem_helper_1(cpath_vocal, index)
        else:  # no caching used
            imitation = self._getitem_helper_2(index)
        if self.gain_augment:
            imitation = pydub_augment(imitation, self.gain_augment)
        if pos_or_neg == 0:  # get negative pair
            recording = self._getitem_helper_3(pos_or_neg, index)
        else:  # == 1 and therefore get positive pair
            recording = self._getitem_helper_3(pos_or_neg, index)
        return imitation, recording, pos_or_neg, self.labels[index]

    def _getitem_helper_1(self, path, index, file="file"):
        try:
            sig = torch.load(path)
        except FileNotFoundError:  # not yet cached
            sig = self._getitem_helper_2(index, file)
            torch.save(sig, path)
        return sig

    def _getitem_helper_2(self, index, file="file"):
        if file == "file":
            sig, _ = librosa.load(os.path.join(dataset_dir, self.files[index]), sr=self.sr, mono=True,
                                  duration=self.dur)
        elif file == "pos_pair":
            sig, _ = librosa.load(os.path.join(dataset_dir, self.positive_pairs[index]), sr=self.sr, mono=True,
                                  duration=self.dur)
        else:  # negative pair
            sig, _ = librosa.load(os.path.join(dataset_dir, self.negative_pairs[index]), sr=self.sr, mono=True,
                                  duration=self.dur)
        sig = self._padding(sig) if self.padding != "no" else sig
        sig = torch.from_numpy(sig[np.newaxis])
        return sig

    def _getitem_helper_3(self, pos_or_neg, index):
        pair = "pos_pair" if pos_or_neg == 1 else "neg_pair"
        if self.cache_path_rec:
            cache_name = self.positive_pairs[index][17:-4] if pos_or_neg == 1 else self.negative_pairs[index][17:-4]
            cpath_rec = os.path.join(self.cache_path_rec, cache_name + ".pt")
            recording = self._getitem_helper_1(cpath_rec, index, file=pair)
        else:  # no caching used
            recording = self._getitem_helper_2(index, file=pair)
        if self.gain_augment:
            recording = pydub_augment(recording, self.gain_augment)
        return recording

    def _padding(self, sig):
        array_length = int(self.dur * self.sr)
        if len(sig) < array_length:
            if self.padding == "zero":  # zero padding
                pad = np.zeros(array_length, dtype="float32")
                pad[:len(sig)] = sig
                sig = pad
            elif self.padding == "conc":  # concatenate
                while len(sig) < array_length:
                    sig = np.concatenate([sig, sig])
                sig = sig[:array_length]
        return sig

    def __len__(self):
        return len(self.files)


class SelectionDataset(Dataset):
    """
    A dataset that selects a subset from a dataset based on a set of sample ids.
    Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset: Dataset, available_indices: List[int], criterion: str = "BCE"):
        self.dataset = dataset
        self.available_indices = available_indices
        self.criterion = criterion

    def __getitem__(self, index):
        if self.criterion == "nt_xent":  # only positive pairs needed
            return self.dataset[(self.available_indices[index], 1)]

        if index % 2 == 0:  # returns a negative pair
            return self.dataset[(self.available_indices[index], 0)]
        else:  # returns a positive pair
            return self.dataset[(self.available_indices[index], 1)]

    def __len__(self):
        return len(self.available_indices)


class MixupDataset(Dataset):
    """
    Mixing Up wave forms
    """

    def __init__(self, dataset: Dataset, beta: float = 2, rate: float = 0.3):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            im1, rec1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            im2, rec2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            im1 = im1 - im1.mean()
            im2 = im2 - im2.mean()
            im = (im1 * l + im2 * (1. - l))
            im = im - im.mean()
            rec1 = rec1 - rec1.mean()
            rec2 = rec2 - rec2.mean()
            rec = (rec1 * l + rec2 * (1. - l))
            rec = rec - rec.mean()
            return im.to(dtype=im1.dtype), rec.to(dtype=rec1.dtype), (y1 * l + y2 * (1. - l)).to(dtype=y1.dtype)
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class PreprocessDataset(Dataset):
    """
    A base preprocessing dataset representing a preprocessing step of a Dataset preprocessed on the fly.
    Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset: Dataset, preprocessor: Callable):
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class RollFunc:
    """
    Roll waveform over time
    """
    def __init__(self, axis: int, shift_range: int):
        self.axis = axis
        self.shift_range = shift_range

    def roll_func(self, batch):
        x = batch[0]  # the waveform
        others = batch[1:]  # label + possible other metadata in batch
        sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        return x.roll(sf, self.axis), *others


def get_roll_func(axis=1, shift_range=4000):
    # roll waveform over time
    roll = RollFunc(axis, shift_range)
    return roll.roll_func


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2 + 1, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


def get_training_set(cache_path="cached", resample_rate=32000, duration=4.0, gain_augment=0, roll=False,
                     mixup_dataset=False, padding="zero", criterion="BCE", fold=0, fine_grained=False):
    set_directory(fine_grained, fold)
    # get filenames of clips in training set
    train_files = pd.read_csv(dataset_config['train_files_csv'], sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    # get indices of training clips
    train_indices = [meta[meta["filename"] == files].index.item() for files in train_files]
    train_indices.sort()
    ds = SelectionDataset(BasicAudioDataset(dataset_config['meta_csv'], sr=resample_rate, duration=duration,
                                            gain_augment=gain_augment, padding=padding, cache_path=cache_path),
                          train_indices,
                          criterion=criterion)
    if roll:  # time shift applied to raw waveforms
        ds = PreprocessDataset(ds, get_roll_func())
    if mixup_dataset:  # mixup applied to raw waveforms
        ds = MixupDataset(ds)
    return ds


def get_val_set(cache_path="cached", resample_rate=32000, duration=4.0, padding="zero", fold=0, fine_grained=False):
    set_directory(fine_grained, fold)
    # get filenames of clips in validation set
    val_files = pd.read_csv(dataset_config['val_files_csv'], sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    # get indices of val clips
    val_indices = [meta[meta["filename"] == files].index.item() for files in val_files]
    val_indices.sort()
    ds = SelectionDataset(BasicAudioDataset(dataset_config['meta_csv'], sr=resample_rate, duration=duration,
                                            padding=padding, cache_path=cache_path),
                          val_indices)
    return ds
