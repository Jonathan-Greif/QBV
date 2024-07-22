import os
import argparse
import numpy as np
import pandas as pd

from helpers.get_files import get_refs_dict
from helpers.get_module import get_module
from helpers.utils_test import get_embeddings_coarse, calc_mrr_coarse


def calculate(config):
    directory = config.directory
    path_ref = config.path_ref
    sr_down = config.sr_down
    dur = config.dur
    arch = config.arch

    print("Own Module" if config.own_module else "Arch: " + str(arch))
    print("SR_down: " + str(sr_down))
    print(f"dur: {dur}")

    # get a dictionary where the keys are sound concepts and the values are their corresponding reference sounds
    references = get_refs_dict(path_ref)

    mrr_folds = []
    fold = 0
    # get encoders for sound recordings and vocal imitations (is the same in some cases)
    # sr_up is the resample rate required for the selected modules
    module_ref, module_im, sr_up = get_module(arch, config.pretrained, config.own_module,
                                              config.state_dict_module, config.state_dict_pretrained, fold)
    # for each fold repeat
    for i in range(10):
        # in this implementation only the own module was trained with data from the folds
        # for the other modules the encoders stay the same
        if config.own_module:
            module_ref, module_im, sr_up = get_module(arch, config.pretrained, config.own_module,
                                                      config.state_dict_module, config.state_dict_pretrained, fold)
        # get the imitations with their labels (concepts) from the current fold
        test_df = pd.read_csv(os.path.join(directory, f"splits/fold{i}/test.csv"), delimiter="\t")
        # select all reference sounds that belong to this fold
        refs_in_test = test_df["scene_label"].unique()
        refs_in_test = [references[x] for x in refs_in_test]
        # get the embeddings of those reference sounds and a sorted list of their concepts
        ref_embs, idx_lst = get_embeddings_coarse(refs_in_test, module_ref, path_ref, arch, sr_down, sr_up, dur)
        # calculate MRR, MeanRecall@1 and MeanReacall@2 for this particular fold
        mrr, r1, r2 = calc_mrr_coarse(test_df, directory, ref_embs, idx_lst, module_im, arch, sr_down, sr_up, dur)

        print(f"\rFold: {fold}, MRR: {round(mrr, 4)}, mR@1: {round(r1, 4)}, mR@2: {round(r2, 4)}",
              end='      ', flush=True)
        mrr_folds.append(mrr)
        fold += 1

    mrr_folds = np.array(mrr_folds)
    print(f"\rMeanReciprocalRank: {round(mrr_folds.mean(), 4)}", end='            \n', flush=True)
    print(f"MRR std: {round(mrr_folds.std(), 4)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # paths
    parser.add_argument('--directory', type=str, default="datasets/data/coarse grained")
    parser.add_argument('--path_ref', type=str,
                        default="datasets/data/coarse grained/sound_recordings")
    parser.add_argument('--state_dict_pretrained', type=tuple,
                        default=(None,
                                 None))
    parser.add_argument('--state_dict_module', type=str,
                        default="resources/ct_nt_xent_fold0mn10d10s32_01.pt")
    # arguments
    parser.add_argument('--arch', type=str, default="MN")  # VGGish, M-VGGish, MN, CQT, 2DFT
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--sr_down', type=int, default=32000)  # 16k, 16k, 32k, 8k, 8k
    parser.add_argument('--dur', type=float, default=10)  # 15.4, 15.4, 10, 15.4, 15.4
    parser.add_argument('--own_module', default=False, action='store_true')

    args = parser.parse_args()

    assert 0 < args.sr_down < 44100, "resample rate negative or too high"
    assert 0 < args.dur <= 15.4, "duration negative or longer than the longest audio file in the dataset"
    assert not (args.pretrained and args.arch != "MN"), "pre-trained with vocal imitations implemented only for MN"

    if args.own_module:
        args.arch = "MN"
        args.sr_down = 32000
        args.dur = 10

    calculate(args)
