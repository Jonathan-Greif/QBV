import torch
import torch.nn as nn

class Dual_Encoder(nn.Module):
    """
    Framework for the dual encoder
    """

    def __init__(self, block1, block2, similarity="cosine", dropout=0.2, single=False):
        super(Dual_Encoder, self).__init__()
        
        self.similarity = similarity
        self.single = single
        
        self.im = block1
        self.rec = block2

        if similarity == "cosine":
            self.sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        else:  # FNN
            self.sim = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2*960, 96),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(96, 1),
                nn.Sigmoid()
            )


    def forward(self, x1, x2, criterion="BCE"):
        if self.single:
            _, x1 = self.im(x1)
            _, x2 = self.im(x2)
        else:
            _, x1 = self.im(x1)
            _, x2 = self.rec(x2)

        if criterion == "nt_xent":
            return x1, x2

        if self.similarity == "cosine":
            x = self.sim(x1, x2)
            x = (x + 1) / 2  # bring similarities in [0,1]
        else:  # FNN
            x = torch.cat((x1, x2), 1)
            x = self.sim(x)

        return x.squeeze().double()



def get_model(block1, block2, similarity="cosine", dropout=0.2, single=False):
    """
    :return: the dual encoder
    """
    dual_encoder = Dual_Encoder(block1, block2, similarity, dropout, single)
    return dual_encoder
