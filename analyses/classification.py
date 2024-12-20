import numpy as np
import scipy.stats as stats
# cite: https://github.com/circstat/pycircstat
# pip install pycircstat
# pip install nose
import pycircstat


# @title @getting_modulation_sig
def getting_modulation_sig(r_np, bins, typename, Idx, co_idx):
    # r_np averaged by bins
    bin_r_np = np.zeros((r_np.shape[0], bins, r_np.shape[2]))
    binlen = int(r_np.shape[1] / bins)
    for ib in range(bins):
        if ib < bins - 1:
            bin_r_np[:, ib, :] = np.mean(
                r_np[:, ib * binlen:(ib + 1) * binlen, :], axis=1)
        if ib == bins - 1:  # avoid binlen too long
            bin_r_np[:, ib, :] = np.mean(r_np[:, ib * binlen:, :], axis=1)

    (SPIdx, MDIdx) = Idx
    nSP = len(np.unique(SPIdx))
    nMD = len(np.unique(MDIdx))
    ic_idx = [i for i in range(nSP) if i != co_idx]

    # gain modulation
    if typename in ['gain', 'all']:
        gain_sig_np = np.zeros((r_np.shape[2],)) - 1
        for iNN in range(r_np.shape[2]):
            gtemp = []
            for isp in range(nSP):
                depthlist = []
                for ib in range(bins):
                    depthi = (
                        np.max(bin_r_np[isp * nMD:(isp + 1) * nMD, ib, iNN])
                        - np.min(bin_r_np[isp * nMD:(isp + 1) * nMD, ib, iNN]))
                    depthlist.append(depthi)
                gtemp.append(depthlist)

            gtemp_np = np.array(gtemp)  # (nSPxbins)

            gtemp_np = gtemp_np - np.tile(gtemp_np[[co_idx], :], (nSP, 1))

            if gtemp_np.max() > 0:
                s1, p1 = stats.wilcoxon(gtemp_np[ic_idx, :].flatten(), 
                                        alternative='two-sided')
                gain_sig_np[iNN] = p1

        if typename == 'gain':
            return gain_sig_np


    # baseline modulation
    if typename in ['baseline', 'all']:
        baseline_sig_np = np.zeros((r_np.shape[2],)) - 1
        for iNN in range(r_np.shape[2]):
            btemp = []
            for isp in range(nSP):
                baselist = []
                for ib in range(bins):
                    basei = np.mean(
                        bin_r_np[isp * nMD:(isp + 1) * nMD, ib, iNN])
                    baselist.append(basei)
                btemp.append(baselist)

            btemp_np = np.array(btemp)  # (nSPxbins)

            btemp_np = btemp_np - np.tile(btemp_np[[co_idx], :], (nSP, 1))

            if btemp_np.max() > 0:
                s2, p2 = stats.wilcoxon(btemp_np[ic_idx, :].flatten(), 
                                        alternative='two-sided')
                baseline_sig_np[iNN] = p2

        if typename == 'baseline':
            return baseline_sig_np


    # vector modualtion
    if typename in ['vector', 'all']:
        mdd = np.arange(0, np.pi * 2, np.pi / 4)
        vector_sig_np = np.zeros((r_np.shape[2],)) - 1
        for iNN in range(r_np.shape[2]):
            vtemp = []
            for isp in range(nSP):
                vectorlist = []
                for ib in range(bins):
                    cossum = np.sum(np.cos(mdd) * (
                        bin_r_np[isp * nMD:(isp + 1) * nMD, ib, iNN]))
                    sinsum = np.sum(np.sin(mdd) * (
                        bin_r_np[isp * nMD:(isp + 1) * nMD, ib, iNN]))
                    vectori = np.arctan2(sinsum, cossum)
                    vectorlist.append(vectori)
                vtemp.append(vectorlist)

            vtemp_np = np.array(vtemp)  # (nSPxbins)

            if vtemp_np.max() > 0:
                strd = 'vtemp_np[0, :]'
                for i in range(nSP - 1):
                    strd += ', vtemp_np[%d, :]' % (i + 1)
                p3, T3 = eval('pycircstat.watson_williams(%s)' % strd)
                # p3, T3 = pycircstat.watson_williams(
                # vtemp_np[ic_idx, :].flatten(),
                # vtemp_np[co_idx, :].flatten())
                vector_sig_np[iNN] = p3

        if typename == 'vector':
            return vector_sig_np

    if typename == 'all':
        return (gain_sig_np, baseline_sig_np, vector_sig_np)
