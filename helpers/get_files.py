import os


def get_file_list(path):
    files = os.listdir(path)
    files.sort()
    return files


def get_imitations_dict(path_im):
    imitations = get_file_list(path_im)
    imitations_dict = dict()
    for i in imitations:
        k = int(i[0:3])
        if k in imitations_dict.keys():
            imitations_dict[k].append(i)
        else:
            imitations_dict[k] = [i]
    return imitations_dict


def get_non_refs_dict(path_non_refs):
    non_references = dict()
    for root, _, files in os.walk(path_non_refs):
        if root == path_non_refs:
            continue
        non_references[int(root[79:82])] = [root, files]
    return non_references


def get_refs_dict(path_refs):
    refs = get_file_list(path_refs)
    references = {}
    for ref in refs:
        references[" ".join(ref[4:8].split("_"))] = ref
    return references
