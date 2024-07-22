import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, GradientAccumulationScheduler

from models.preprocess import AugmentMelSTFT
from models.mn.model import get_model as get_mobilenet
from models.dual_encoder import get_model

from datasets.audiodataset import get_val_set, get_training_set
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, mixup, mixstyle, NTXent
from helpers.lr_schedule import exp_warmup_linear_down


class DualEncoder(pl.LightningModule):
    """
    Pytorch Lightning Module to fine-tune the to be specified dual encoder for either coarse-grained or
    fine-grained QBV. Encoders based on MobileNetV3.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # criterion used to fine-tune the dual encoder for QBV
        if config.criterion == "nt_xent":
            self.criterion = NTXent()
        else:  # BCE
            self.criterion = None

        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )

        # get the to be specified mobilenetV3 as encoder
        pretrained_name = config.pretrained_name
        width = NAME_TO_WIDTH(pretrained_name)
        self.block1 = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                                    head_type=config.head_type, se_dims=config.se_dims, num_classes=config.n_classes)

        single = config.single
        if single:  # the same encoder for both domains (shared weights)
            self.block2 = None
        else:  # dual encoder
            self.block2 = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                                        head_type=config.head_type, se_dims=config.se_dims,
                                        num_classes=config.n_classes)

        if config.pretrained:  # supervised pre-training with vocal imitations
            pretrained_dict_b1 = torch.load(config.path_state_dict[0])
            block1_dict = {k: pretrained_dict_b1[k] for k, _ in self.block1.state_dict().items() if
                           k in pretrained_dict_b1}
            self.block1.load_state_dict(block1_dict)
            if not single:
                pretrained_dict_b2 = torch.load(config.path_state_dict[1])
                block2_dict = {k: pretrained_dict_b2[k] for k, _ in self.block2.state_dict().items() if
                               k in pretrained_dict_b2}
                self.block2.load_state_dict(block2_dict)
            print("Supervised pre-training with vocal imitations is used. \n",
                  f"Imitation-Encoder: {config.path_state_dict[0]}\n",
                  f"Recording-Encoder: {config.path_state_dict[1]}")

        self.model = get_model(self.block1, self.block2, config.similarity, dropout=config.dropout, single=single)

    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating mel spectrograms we remove the channel dimension
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    def forward(self, x1, x2):
        """
        :param x1: batch of raw vocal imitations (waveforms)
        :param x2: batch of raw sound recordings (waveforms)
        :return: final model predictions
        """
        x1 = self.mel_forward(x1)
        x2 = self.mel_forward(x2)
        y_hat = self.model(x1, x2)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        imitation, recording, y, labels = train_batch
        bs = imitation.size(0)

        imitation, recording = [self.mel_forward(x) for x in [imitation, recording]]

        if self.config.mixstyle_p > 0:
            imitation = mixstyle(imitation, self.config.mixstyle_p, self.config.mixstyle_alpha)
            recording = mixstyle(recording, self.config.mixstyle_p, self.config.mixstyle_alpha)

        if self.config.mixup_alpha:
            rn_indices, lam = mixup(bs, self.config.mixup_alpha)  # get shuffled indices and mixing coefficients
            lam = lam.to(imitation.device).reshape(bs, 1, 1, 1)
            imitation = imitation * lam + imitation[rn_indices] * (1. - lam)
            recording = recording * lam + recording[rn_indices] * (1. - lam)
            if self.criterion:
                im_embs, rec_embs = self.model(imitation, recording, "nt_xent")
                samples_loss = (
                    self.criterion(im_embs, rec_embs, (labels, labels[rn_indices]), mixup=True)
                )
            else:
                y_hat = self.model(imitation, recording)
                samples_loss = (
                        F.binary_cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                        F.binary_cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(bs))
                )
        else:
            if self.criterion:
                im_embs, rec_embs = self.model(imitation, recording, "nt_xent")
                samples_loss = (
                    self.criterion(im_embs, rec_embs, labels)
                )
            else:
                y_hat = self.model(imitation, recording)
                samples_loss = F.binary_cross_entropy(y_hat.float(), y.float(), reduction="none")

        loss = samples_loss.mean()
        results = {"loss": loss}
        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_dict({'loss': avg_loss})

    def validation_step(self, train_batch, batch_idx):
        imitation, recording, y, _ = train_batch
        imitation, recording = [self.mel_forward(x) for x in [imitation, recording]]

        y_hat = self.model(imitation, recording)
        samples_loss = F.binary_cross_entropy(y_hat.float(), y.float())

        results = {"val_loss": samples_loss}
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log_dict({'val_loss': avg_loss})


def train(config):
    # Train dual encoders for QBV

    pretrained = "pre" if config.pretrained else ""
    if config.fine_grained:
        ID = (f"ct_fine_{config.criterion}_" + config.pretrained_name[:4] + f"d{int(config.duration)}" +
              f"s{int(config.resample_rate / 1000)}{pretrained}_" + str(config.id))
    else:
        ID = (f"ct_{config.criterion}_fold{config.fold}" + config.pretrained_name[:4] + f"d{int(config.duration)}" +
              f"s{int(config.resample_rate/1000)}{pretrained}_" + str(config.id))

    wandb_logger = WandbLogger(
        project=config.project,
        notes="Pipeline for QBV",
        tags=["VII"],
        config=config,
        name=config.pretrained_name + " lr=" + str(config.lr) + " wd=" + str(config.weight_decay) +
            f" mixupalpha={config.mixup_alpha}" + f" mixstylep={config.mixstyle_p}" + f" pretrained={config.pretrained}"
            + " similarity=" + config.similarity + f" id={ID}"
    )

    train_dl = DataLoader(dataset=get_training_set(config.cache_path, config.resample_rate, config.duration,
                                                   config.gain_augment, config.roll, config.mixup_dataset,
                                                   config.padding, config.criterion, config.fold, config.fine_grained),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    val_dl = DataLoader(dataset=get_val_set(config.cache_path, config.resample_rate, config.duration,
                                            config.padding, config.fold, config.fine_grained),
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    pl_module = DualEncoder(config)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accumulator = GradientAccumulationScheduler(scheduling={0: 2})

    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=config.num_gpus,
                         callbacks=[lr_monitor, accumulator],
                         )

    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if config.save_model:  # save the trained dual encoder
        path = "resources/" + ID + ".pt"
        torch.save(pl_module.model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project', type=str, default="QBV")
    parser.add_argument('--experiment_name', type=str, default="mobilenet")
    parser.add_argument('--id', type=str, default="001")
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--save_model', default=False, action='store_true')

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="cached")
    parser.add_argument('--fine_grained', default=False, action='store_true')
    parser.add_argument('--fold', type=int, default=2)

    # Encoder
    parser.add_argument('--n_classes', type=int, default=476)
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the encoder
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--channels_multiplier', type=int, default=2)
    parser.add_argument('--pretrained_name', type=str, default="mn10_as")
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")

    parser.add_argument('--single', default=False, action='store_true')
    parser.add_argument('--similarity', type=str, default="cosine")  # cosine or FNN
    parser.add_argument('--pretrained', default=False, action="store_true")
    parser.add_argument('--path_state_dict', type=tuple,
                        default=("resources/VocalSketch120_mn10d10s32_320.pt",
                                 "resources/VocalSketch120_mn10d10s32_320.pt"))
    parser.add_argument('--criterion', type=str, default="nt_xent")  # nt_xent or BCE

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--mixup_alpha', type=float, default=0.)
    parser.add_argument('--mixstyle_p', type=float, default=0.3)
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--weight_loss', type=float, default=1.)
    parser.add_argument('--dropout', type=float, default=0.2)

    # learning rate + schedule
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--warm_up_len', type=int, default=2)
    parser.add_argument('--ramp_down_start', type=int, default=4)
    parser.add_argument('--ramp_down_len', type=int, default=7)
    parser.add_argument('--last_lr_value', type=float, default=0.0001)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--padding', type=str, default="zero")  # no padding with "no", zero with "zero",
                                                                             # concatenated with "conc"
    parser.add_argument('--gain_augment', type=int, default=0)
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time
    parser.add_argument('--mixup_dataset', default=False, action='store_true')

    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=2)  # frequency masking
    parser.add_argument('--timem', type=int, default=400)  # time masking
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(1234)
    np.random.seed(1234)

    train(args)
