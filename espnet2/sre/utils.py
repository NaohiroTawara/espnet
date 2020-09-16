from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np
from sklearn import metrics
import torch


def read_utt2spk(utt2spk: Union[str, Path]):
    utt2spk_dict = {}
    spk2utt_dict = {}
    with open(utt2spk, encoding="utf-8") as f:
        for line in f:
            utt, *spk = line.split()
            if len(spk) != 1:
                raise RuntimeError(f"Format error: {line}")
            if utt in utt2spk_dict:
                raise RuntimeError(f"Duplicated ID: {utt}")
            spk = spk[0]
            utt2spk_dict[utt] = spk
            spk2utt_dict.setdefault(spk, []).append(utt)

    if len(utt2spk_dict) == 0:
        raise RuntimeError(f"Empty file: {utt2spk}")
    spk2spkid = {}
    spkid2spk = {}
    for idx, spk in enumerate(spk2utt_dict):
        spk2spkid[spk] = idx
        spkid2spk[idx] = spk
    return utt2spk_dict, spk2utt_dict, spk2spkid, spkid2spk


def calculate_eer(labels, scores) -> Tuple[float, float, np.array, np.array, float]:
    idx = torch.where(labels>=0)
    fpr, tpr, thresholds = metrics.roc_curve(labels[idx], scores[idx], pos_label=1)
    fnr = 1 - tpr

    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idx], fnr[idx])
    auc = metrics.auc(fpr, tpr)
    return eer, thresholds[idx].astype(np.float64), fpr, tpr, auc


def getTunedThreshold(
        labels: torch.Tensor,
        scores: torch.Tensor,
):
    """ Get thresholds that satisfies target_fa, target_fr and EER"""
    assert len(labels) == len(scores)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    fnr *= 100
    fpr *= 100
    tuned_thresholds = []
    '''
    for target_fa in target_fas:
        idx = np.nanargmin(np.absolute(target_fa - fpr))
        tuned_thresholds.append([thresholds[idx], fpr[idx], fnr[idx]])
    '''
    idxE = np.nanargmin(np.absolute(fnr - fpr)) # obtain the nearest threshold that satisfies fnr == fpr
    eer = max(fpr[idxE], fnr[idxE])
    tuned_thresholds.append(thresholds[idxE])
    return (tuned_thresholds, eer, fpr, fnr)
