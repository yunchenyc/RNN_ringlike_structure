import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist


def getting_neural_states_withPCA(r_np, t_left, t_right, time_marker, bins, cd):
    r_cut = np.zeros((r_np.shape[0], int(t_right - t_left), r_np.shape[2]))
    for itrial in range(r_np.shape[0]):
        r_cut[itrial, :, :] = \
            r_np[itrial,
                 cd.time_dict[time_marker][itrial] + t_left
                 :cd.time_dict[time_marker][itrial] + t_right, :]

    # r_np averaged by bins
    bin_r_np = np.zeros((r_cut.shape[0], bins, r_cut.shape[2]))
    binlen = int(r_cut.shape[1] / bins)
    for ib in range(bins):
        if ib < bins - 1:
            bin_r_np[:, ib, :] = np.mean(
                r_cut[:, ib * binlen:(ib + 1) * binlen, :], axis=1)
        if ib == bins - 1:  # avoid binlen too long
            bin_r_np[:, ib, :] = np.mean(r_cut[:, ib * binlen:, :], axis=1)

    r_c_nt = bin_r_np.reshape((r_np.shape[0], bins * r_np.shape[2]))

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(r_c_nt)
    evr = pca.explained_variance_ratio_

    return pcs, evr


def calculating_distance_by_group(pcs, idx, method='euclidean'):
    pcs_all = [(pcs[i, 0], pcs[i, 1], pcs[i, 2]) for i in range(pcs.shape[0])]
    distmax = (pdist(pcs_all, method)).max()

    within_group_distance = []
    for igroup in np.unique(idx):
        pcs_points = (pcs[idx==igroup, 0], 
                      pcs[idx==igroup, 1], 
                      pcs[idx==igroup, 2])
            
        dist_mdcluster = pdist(pcs_points, method)/distmax
        within_group_distance.append([np.mean(dist_mdcluster), np.std(dist_mdcluster)])
    
    return within_group_distance
