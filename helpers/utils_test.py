import os
import numpy as np
import torch
import librosa
from helpers.cqt.cqt import cqt


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_single_emb(module, arch, file):
    if arch == "VGGish":
        file = np.array(file)
        emb = module(file, 16000).view(-1)
    elif arch == "M-VGGish":
        file = np.array(file)
        emb = module(file)
    elif arch == "MN":
        file = file.unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
        emb = module(file)
        emb = emb.squeeze().to("cpu")
    elif arch == "CQT":
        emb = torch.tensor(np.abs(file).reshape(-1))
    else:  # 2DFT
        file = calculate_2dft(file)
        emb = torch.tensor(np.abs(file).reshape(-1))
    return emb


def calculate_2dft(x):
    ft = np.abs(x)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def padding(file, sr, dur):
    array_length = int(sr * dur)
    if len(file) < array_length:
        container = np.zeros(array_length)
        container[:len(file)] = file
        file = container
    else:
        file = file[:array_length]
    return file


def get_file(file, path, arch, sr_down=8000, sr_up=8000, dur=15.4):
    file_path = os.path.join(path, file)
    # the following two lines of code were chosen in alignment with:
    # https://archive.nyu.edu/bitstream/2451/60758/1/DCASE2019Workshop_Pishdadian_51.pdf (1)
    file, _ = librosa.load(file_path, sr=sr_down, mono=True)
    file = librosa.resample(file, orig_sr=sr_down, target_sr=sr_up)
    file = padding(file, sr_up, dur)
    if arch == "CQT" or arch == "2DFT":
        c = cqt(file, 12, sr_up, 55, 2090)
        # best parameter settings were chosen in alignment with (1)
        file = c["cqt"]
    else:
        file = torch.from_numpy(file)
    return file


def get_embeddings_coarse(files, module, path, arch, sr_down, sr_up, dur):
    with torch.no_grad():
        file = get_file(files[0], path, arch, sr_down, sr_up, dur)
        emb = get_single_emb(module, arch, file)
        matrix = torch.zeros((len(files), emb.shape[0]))
        idx_lst = [0]*len(files)
        matrix[0] = emb
        idx_lst[0] = files[0][4:8]
        for i in range(1, len(files)):
            file = get_file(files[i], path, arch, sr_down, sr_up, dur)
            matrix[i] = get_single_emb(module, arch, file)
            idx_lst[i] = files[i][4:8]
    return matrix, idx_lst


def get_embeddings_fine(ref, non_refs, path_ref, module, arch, sr_down, sr_up, dur):
    with torch.no_grad():
        file = get_file(ref, path_ref, arch, sr_down, sr_up, dur)
        emb = get_single_emb(module, arch, file)
        path = non_refs[0]
        files = non_refs[1]
        matrix = torch.zeros((len(files)+1, emb.shape[0]))
        matrix[0] = emb
        for i, file in enumerate(files):
            file = get_file(file, path, arch, sr_down, sr_up, dur)
            matrix[i+1] = get_single_emb(module, arch, file)
    return matrix


def calc_mrr_coarse(df, path, ref_embs, idx_lst, module, arch, sr_down, sr_up, dur):
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    mrr_lst = []
    rank1, rank2 = 0, 0
    with torch.no_grad():
        for i, row in df.iterrows():
            filename, label = row[1], row[2]
            im = get_file(filename, path, arch, sr_down, sr_up, dur)
            emb = get_single_emb(module, arch, im).to("cpu")
            cos_sim = sim(emb, ref_embs)
            position = idx_lst.index(label)
            sorted_indices = torch.sort(cos_sim, descending=True)[1]
            rank = (sorted_indices == position).nonzero(as_tuple=True)[0].item()
            rank += 1
            if rank == 1:
                rank1 += 1
            elif rank == 2:
                rank2 += 1
            mrr_lst.append(1 / rank)
    mrr = np.array(mrr_lst)
    r1 = rank1/len(df)
    r2 = (rank1 + rank2)/len(df)
    return mrr.mean(), r1, r2


def calc_mrr_fine(imitations, rec_embs, module, path, arch, sr_down, sr_up, dur, rank1, rank2):
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    mrr_lst = []
    with torch.no_grad():
        for imitation in imitations:
            im = get_file(imitation, path, arch, sr_down, sr_up, dur)
            emb = get_single_emb(module, arch, im).to("cpu")
            cos_sim = sim(emb, rec_embs)
            position = 0
            sorted_indices = torch.sort(cos_sim, descending=True)[1]
            rank = (sorted_indices == position).nonzero(as_tuple=True)[0].item()
            rank += 1
            if rank == 1:
                rank1 += 1
            if rank == 2:
                rank2 += 1
            mrr_lst.append(1 / rank)
    mrr = np.array(mrr_lst)
    return mrr.mean(), rank1, rank2
