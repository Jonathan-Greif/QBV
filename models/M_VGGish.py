import numpy as np
import torch
import torch.nn as nn


class M_VGGish(nn.Module):
    """
    2s in the style of: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683461
    """
    def __init__(self, sr=16000):
        super(M_VGGish, self).__init__()
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.postprocess = False
        self.l5 = self.model.features[:12]
        self.l6 = self.model.features[:14]

        self.sr = sr  # sampling rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        two_sec_segments, x = self.get_number_of_2s_segments(x)

        x = self.model._preprocess(x, self.sr).to(self.device)
        x = x.reshape(two_sec_segments, 1, -1, x.shape[3])
        
        x1 = self.l5(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = x1.reshape(two_sec_segments, -1)
        
        x2 = self.l6(x)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = x2.reshape(two_sec_segments, -1)
        
        x = torch.cat((x1, x2), dim=1)
        x = x.mean(dim=0)
        return x

    def get_number_of_2s_segments(self, x):
        n_segments = np.ceil((len(x) / self.sr))
        if n_segments % 2 != 0:
            container = np.zeros(len(x) + self.sr)
            container[:len(x)] = x
            x = container
            n_segments += 1
        return int(n_segments / 2), x
