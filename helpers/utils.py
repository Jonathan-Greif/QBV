import torch
import torch.nn as nn
import numpy as np
import librosa
from torch.distributions.beta import Beta


def NAME_TO_WIDTH(name):
    map = {
        'mn01': 0.1,
        'mn02': 0.2,
        'mn04': 0.4,
        'mn05': 0.5,
        'mn06': 0.6,
        'mn08': 0.8,
        'mn10': 1.0,
        'mn12': 1.2,
        'mn14': 1.4,
        'mn16': 1.6,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0
    }
    try:
        w = map[name[:4]]
    except:
        w = 1.0

    return w


def mixup(size, alpha):
    # https://arxiv.org/abs/1710.09412
    rn_indices = torch.randperm(size)  # randomly shuffles batch indices
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)  # choose mixing coefficients from Beta distribution
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)  # choose lambda closer to 1
    lam = torch.FloatTensor(lambd)  # convert to pytorch float tensor
    return rn_indices, lam


def mixstyle(x, p=0.4, alpha=0.4, eps=1e-6, mix_labels=False):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics
    if mix_labels:
        return x, perm, lmda
    return x


def wav_to_torch(path, sr=32000, dur=10):
    """
    From given path convert a WAV file into a tensor of an audio sample
    :return: tensor of an audio sample
    """
    sig, _ = librosa.load(path, sr=sr, mono=True,
                          duration=dur)

    # create zero padding if needed
    array_length = int(dur * sr)
    if len(sig) < array_length:
        pad = np.zeros(array_length, dtype="float32")
        pad[:len(sig)] = sig
        sig = pad

    sig = torch.from_numpy(sig[np.newaxis])
    return sig.unsqueeze(0)


class NTXent(nn.Module):
    """
    Adapted NT-Xent loss for supervised cross-domain retrieval task.
    Taken and further adapted from here: https://github.com/XinhaoMei/audio-text_retrieval/blob/main/tools/loss.py
    """

    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=0)
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.tau = temperature

    def forward(self, imitation_embeds, recording_embeds, labels, mxup=False):

        n = imitation_embeds.shape[0]
        i2r = torch.stack([self.sim(im, recording_embeds) for im in imitation_embeds]) / self.tau

        if mxup:
            labels1, labels2 = labels
            mask = [0 if (labels1[i] in (labels1[k], labels2[k]) or labels2[i] in (labels1[k], labels2[k])) and k != i
                    else 1 for i in range(n) for k in range(n)]
            mask = np.array(mask).reshape(n, n)
        else:
            mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(i2r.device)
            mask_diag = mask.diag()
            mask_diag = torch.diag_embed(mask_diag)
            mask = ~(mask ^ mask_diag)

        loss_i2r = [self.loss(i2r[i][mask[i]])[sum(mask[i][:i])] for i in range(n)]
        loss = ((- sum(loss_i2r)) / len(loss_i2r))
        return loss
