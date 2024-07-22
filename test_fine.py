import argparse
import numpy as np

from helpers.get_files import get_file_list, get_non_refs_dict, get_imitations_dict
from helpers.get_module import get_module
from helpers.utils_test import get_embeddings_fine, calc_mrr_fine


def calculate(config):
    path_ref = config.path_ref
    path_non_refs = config.path_non_refs
    path_im = config.path_im
    sr_down = config.sr_down
    dur = config.dur
    arch = config.arch

    # get all reference sounds
    all_refs = get_file_list(path_ref)
    # get a dictionary where the keys are sound concepts and the values are their corresponding non reference sounds
    all_non_refs = get_non_refs_dict(path_non_refs)
    # get a dictionary where the keys are sound concepts and the values are their corresponding vocal imitations
    all_imitations = get_imitations_dict(path_im)

    # get encoders for sound recordings and vocal imitations (is the same in some cases)
    # sr_up is the resample rate required for the selected modules
    module_ref, module_im, sr_up = get_module(arch, config.pretrained, config.own_module,
                                              config.state_dict_module, config.state_dict_pretrained)
    print("Own Module" if config.own_module else "Arch: " + str(arch))
    print("SR_down: " + str(sr_down))
    print(f"dur: {dur}")

    mrr_lst = []
    rank1 = 0
    rank2 = 0

    # Iterate over the different concepts and calculate MRR, MR@1 and MR@2
    for key in all_imitations.keys():
        if key == 239:  # there don't exist non reference sounds for this concept
            continue
        # get the embeddings of the reference sound and of all non reference sounds from this concept (~10 in total)
        refs_emb = get_embeddings_fine(all_refs[key], all_non_refs[key], path_ref, module_ref, arch,
                                       sr_down, sr_up, dur)
        # calculate MRR for the current concept. rank1 and rank2 are counter and get evaluated at the end.
        mrr, rank1, rank2 = calc_mrr_fine(all_imitations[key], refs_emb, module_im, path_im, arch, sr_down, sr_up, dur,
                                          rank1, rank2)
        print(f"\rkey: {key}, MRR: {round(mrr,4)}", end='    ', flush=True)
        mrr_lst.append(mrr)

    mrr_lst = np.array(mrr_lst)
    print(f"\rMeanReciprocalRank: {round(mrr_lst.mean(), 4)}", end='       \n', flush=True)
    n_queries = len([file for lst in all_imitations.values() for file in lst])
    print(f"MeanRecall@1: {round(rank1/n_queries, 4)}")
    print(f"MeanRecall@2: {round((rank1 + rank2)/n_queries, 4)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # paths
    parser.add_argument('--path_im', type=str,
                        default="datasets/data/fine grained/VocalImitationSet/vocal_imitations")
    parser.add_argument('--path_ref', type=str,
                        default="datasets/data/fine grained/VocalImitationSet/original_recordings/reference")
    parser.add_argument('--path_non_refs', type=str,
                        default="datasets/data/fine grained/VocalImitationSet/original_recordings/non_reference")
    parser.add_argument('--state_dict_pretrained', type=tuple,
                        default=("resources/VocalSketch120_mn10d10s32_320.pt",
                                 "resources/VocalSketch120_mn10d10s32_320.pt"))
    parser.add_argument('--state_dict_module', type=str,
                        default="resources/ct_fine_nt_xent_mn10d10s32_01.pt")
    # arguments
    parser.add_argument('--arch', type=str, default="MN")  # VGGish, M-VGGish, MN, CQT, 2DFT
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--sr_down', type=int, default=32000)  # 16k, 16k, 32k, 8k, 8k
    parser.add_argument('--dur', type=float, default=10)  # 15.4, 15.4, 10, 15.4, 15.4
    parser.add_argument('--own_module', default=False, action='store_true')

    args = parser.parse_args()

    assert 0 < args.sr_down < 44100, "resample rate negative or too high"
    assert 0 < args.dur <= 15.4, "duration negative or too long"
    assert not (args.pretrained and args.arch != "MN"), "pre-trained with vocal imitations implemented only for MN"

    if args.own_module:
        args.arch = "MN"
        args.sr_down = 32000
        args.dur = 10

    calculate(args)
