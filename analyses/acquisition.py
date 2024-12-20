import numpy as np


def getting_nonzero_rate_slice(rate_np, Idx, cd, period, timemarker):
    tl, tr = period
    nTrial, nNeuron = rate_np.shape[0], rate_np.shape[2]
    rate_cut = np.zeros((nTrial, int(tr - tl), nNeuron))
    if timemarker in ['go', 'mo']:
        for iTrial in range(nTrial):
            rate_cut[iTrial, :, :] = (
                rate_np[iTrial, 
                        cd.time_dict[timemarker][iTrial] + tl
                        :cd.time_dict[timemarker][iTrial] + tr, :])
    elif timemarker == 'to':
        for iTrial in range(nTrial):
            rate_cut[iTrial, :, :] = (rate_np[iTrial, tl:tr, :])

    # trial-averaged
    (SPIdx, MDIdx) = Idx
    nSP = len(np.unique(SPIdx))
    nMD = len(np.unique(MDIdx))
    rate_mean_all = np.zeros((nSP * nMD, int(tr - tl), nNeuron))
    for iSP in range(nSP):
        for iMD in range(nMD):
            rate_mean_all[iSP * nMD + iMD, :, :] = \
                np.mean(rate_cut[(SPIdx == iSP) & (MDIdx == iMD), :, :], 
                        axis=0)

    # nonzero
    nonzero_idx = []
    for iN in range(nNeuron):
        if rate_mean_all[:, :, iN].max() > 0:
            nonzero_idx.append(iN)
    rate_nz = rate_mean_all[:, :, nonzero_idx]

    return rate_nz, nonzero_idx


def getting_nonzero_rate_slice_trial(rate_np, cd, period, timemarker):
    tl, tr = period
    nTrial, nNeuron = rate_np.shape[0], rate_np.shape[2]
    rate_cut = np.zeros((nTrial, int(tr - tl), nNeuron))
    if timemarker in ['go', 'mo']:
        for iTrial in range(nTrial):
            rate_cut[iTrial, :, :] = (
                rate_np[iTrial,
                        cd.time_dict[timemarker][iTrial] + tl
                        :cd.time_dict[timemarker][iTrial] + tr, 
                        :])
    elif timemarker == 'to':
        for iTrial in range(nTrial):
            rate_cut[iTrial, :, :] = (rate_np[iTrial, tl:tr, :])

    # nonzero
    nonzero_idx = []
    for iN in range(nNeuron):
        if rate_cut[:, :, iN].max() > 0:
            nonzero_idx.append(iN)
    rate_nz = rate_cut[:, :, nonzero_idx]

    return rate_nz, nonzero_idx


# @title @getting_rate_slice

def getting_rate_slice(rate_np, cd, period, timemarker, idxlist=None):
    tl, tr = period
    nTrial, nNeuron = rate_np.shape[0], rate_np.shape[2]
    rate_cut = np.zeros((nTrial, int(tr - tl), nNeuron))
    if timemarker in ['go', 'mo']:
        for iTrial in range(nTrial):
            rate_cut[iTrial, :, :] = (
                rate_np[iTrial,
                        cd.time_dict[timemarker][iTrial] + tl
                        :cd.time_dict[timemarker][iTrial] + tr, 
                        :])
    elif timemarker == 'to':
        for iTrial in range(nTrial):
            rate_cut[iTrial, :, :] = (rate_np[iTrial, tl:tr, :])

    if idxlist is None:
        return rate_cut
    else:
        return rate_cut[:, :, idxlist]


