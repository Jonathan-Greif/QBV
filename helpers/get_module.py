import re
import torch
from .. import ex_qbv
from ..models.M_VGGish import M_VGGish


device = "cuda" if torch.cuda.is_available() else "cpu"


def replace_fold_number(state_dict_module, fold):
    pattern = r"(fold)\d+"
    replacement = f"fold{fold}"
    result = re.sub(pattern, replacement, state_dict_module)
    return result


class Encoder(torch.nn.Module):
    def __init__(self, module, encoder):
        super(Encoder, self).__init__()
        self.mel = module.mel
        self.pre = module.mel_forward
        self.encoder = encoder

    def forward(self, x):
        x = self.pre(x)
        _, x = self.encoder(x)
        return x


class Config_QBV:
    def __init__(self, width=1.0, sim="cosine", single=False, pretrained=False, 
    		 state_dict_pretrained=(None, None), fold=-1):
        assert not (pretrained and state_dict_pretrained is None), \
            "Pretrained is True but state_dict_pretrained is None"
        # model
        width = int(width * 10)
        width = f"0{width}" if width < 10 else width
        self.pretrained_name = f"mn{width}_as"
        self.model_width = width
        self.head_type = "mlp"
        self.se_dims = "c"
        self.n_classes = 476 if fold >= 0 else 120
        self.pretrained = pretrained
        self.path_state_dict = state_dict_pretrained
        self.similarity = sim
        self.single = single
        self.criterion = "BCE"
        self.dropout = 0.2
        # preprocessing
        self.resample_rate = 32000
        self.duration = 10.0
        self.pretrain_final_temp = 1.0
        self.window_size = 800
        self.hop_size = 320
        self.n_fft = 1024
        self.n_mels = 128
        self.freqm = 0
        self.timem = 0
        self.fmin = 0
        self.fmax = None
        self.fmin_aug_range = 10
        self.fmax_aug_range = 2000


def get_module(arch: str, pretrained: bool, own: bool, state_dict_module: str, state_dict_pretrained: tuple,
               fold: int = -1):
    if own:
        conf = Config_QBV(sim="cosine", fold=fold)
        module = ex_qbv.DualEncoder(conf)
        if fold >= 0:
            state_dict_module = replace_fold_number(state_dict_module, fold)
            pretrained_dict = torch.load(state_dict_module)
        else:
            pretrained_dict = torch.load(state_dict_module)
        model_dict = {k: pretrained_dict[k] for k, _ in module.model.state_dict().items() if
                      k in pretrained_dict}
        module.model.load_state_dict(model_dict)
        print("Pretrained: " + state_dict_module)
        module_im = Encoder(module, module.block1)
        module_ref = Encoder(module, module.block2)
        for model in [module_im, module_ref]:
            model.to(device)
            model.eval()
        return module_ref, module_im, 32000

    if arch == "VGGish":
        model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        model.postprocess = False
        model.to(device)
        model.eval()
        return model, model, 16000

    elif arch == "M-VGGish":
        model = M_VGGish(sr=16000)
        model.to(device)
        model.eval()
        return model, model, 16000

    elif arch == "MN":
        conf = Config_QBV(sim="cosine", pretrained=pretrained, state_dict_pretrained=state_dict_pretrained)
        module = ex_qbv.DualEncoder(conf)
        module_im = Encoder(module, module.block1)
        module_ref = Encoder(module, module.block2)
        for model in [module_im, module_ref]:
            model.to(device)
            model.eval()
        return module_ref, module_im, 32000

    elif arch == "CQT" or arch == "2DFT":
        return None, None, 8000

    else:
        raise AssertionError("Invalid Module(arch)")
