# -*- coding: utf-8 -*-

#%%
import os
import torch
import numpy as np
import pandas as pd
import settings.RNN_setting as rs
import settings.task_setting as ts
import analyses.acquisition as ac
import helpers.helper_functions as helper
from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.spatial import procrustes
    

### Table S2 ###

#%% neural data

# basic setup
PATH_LOAD_data = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/monkey_data')

### Monkey C
# load data
m1_data = loadmat(os.path.join(PATH_LOAD_data, 'C_1022_M1.mat'))
pmd_data = loadmat(os.path.join(PATH_LOAD_data, 'C_1022_PMd.mat'))
m_rate_to = m1_data['FR_TO']
m_rate_mo = m1_data['FR_MO']
m_rate_mo2 = pmd_data['FR_MO']

# PCA
pca = PCA(n_components=30)

# MO 1-300 trials
m11 = m_rate_mo[:300, :, :]
r_ct1 = m11.reshape(m11.shape[0]*m11.shape[1], m11.shape[2])
pcs1 = pca.fit_transform(r_ct1)
evr1 = pca.explained_variance_ratio_

# TO 1-300 trials
m22 = m_rate_to[:300, :, :]
r_ct2 = m22.reshape(m22.shape[0]*m22.shape[1], m22.shape[2])
pcs2 = pca.fit_transform(r_ct2)
evr2 = pca.explained_variance_ratio_

# MO 300-600 trials
m33 = m_rate_mo[300:600, :, :]
r_ct3 = m33.reshape(m33.shape[0]*m33.shape[1], m33.shape[2])
pcs3 = pca.fit_transform(r_ct3)
evr3 = pca.explained_variance_ratio_

# PMd 1-300 trials
m44 = m_rate_mo2[:300, :, :]
r_ct4 = m44.reshape(m44.shape[0]*m44.shape[1], m44.shape[2])
pcs4 = pca.fit_transform(r_ct4)
evr4 = pca.explained_variance_ratio_

# CCA
cca = CCA(n_components=10)

# MO vs TO
cca.fit(pcs1, pcs2)
X_c, Y_c = cca.transform(pcs1, pcs2)
pcs12_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
mtx1, mtx2, pcs12_disparity = procrustes(pcs1, pcs2)
print('C-M1-MO vs. C-M1-TO')
print('CC1 = %.2f, CC2 = %.2f, CC3 = %.2f, Disparity = %.2f' %
      (pcs12_correlations[0], pcs12_correlations[1], pcs12_correlations[2], pcs12_disparity))

# trial 1-300 vs trial 301-600
cca.fit(pcs1, pcs3)
X_c, Y_c = cca.transform(pcs1, pcs3)
pcs13_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
mtx1, mtx2, pcs13_disparity = procrustes(pcs1, pcs3)
print('C-M1-MO-1 vs. C-M1-MO-2')
print('CC1 = %.2f, CC2 = %.2f, CC3 = %.2f, Disparity = %.2f' %
      (pcs13_correlations[0], pcs13_correlations[1], pcs13_correlations[2], pcs13_disparity))

# M1 vs PMd
cca.fit(pcs1, pcs4)
X_c, Y_c = cca.transform(pcs1, pcs4)
pcs14_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
mtx1, mtx2, pcs14_disparity = procrustes(pcs1, pcs4)
print('C-M1-MO vs. C-PMd-MO')
print('CC1 = %.2f, CC2 = %.2f, CC3 = %.2f, Disparity = %.2f' %
      (pcs14_correlations[0], pcs14_correlations[1], pcs14_correlations[2], pcs14_disparity))

### Monkey G
m1_data_G1 = loadmat(os.path.join(PATH_LOAD_data, 'G_0914_M1.mat'))
m1_data_G2 = loadmat(os.path.join(PATH_LOAD_data, 'G_0915_M1.mat'))

# PCA
pca = PCA(n_components=30)

# MO 1-300 trials G session1
mG1 = m1_data_G1['Fr_MO'][:300, :, :]
r_ctG1 = mG1.reshape(mG1.shape[0]*mG1.shape[1], mG1.shape[2])
pcsG1 = pca.fit_transform(r_ctG1)
evrG1 = pca.explained_variance_ratio_

# MO 1-300 trials G session2
mG2 = m1_data_G2['Fr_MO'][:300, :, :]
r_ctG2 = mG2.reshape(mG2.shape[0]*mG2.shape[1], mG2.shape[2])
pcsG2 = pca.fit_transform(r_ctG2)
evrG2 = pca.explained_variance_ratio_

# C1022M1 vs G0914M1
cca.fit(pcs1, pcsG1)
X_c, Y_c = cca.transform(pcs1, pcsG1)
pcs1G1_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
mtx1, mtx2, pcs1G1_disparity = procrustes(pcs1, pcsG1)
print('C-M1-MO vs. G-M1-MO-1')
print('CC1 = %.2f, CC2 = %.2f, CC3 = %.2f, Disparity = %.2f' %
      (pcs1G1_correlations[0], pcs1G1_correlations[1], pcs1G1_correlations[2], pcs1G1_disparity))

# C1022M1 vs G0915M1
cca.fit(pcs1, pcsG2)
X_c, Y_c = cca.transform(pcs1, pcsG2)
pcs1G2_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
mtx1, mtx2, pcs1G2_disparity = procrustes(pcs1, pcsG2)
print('C-M1-MO vs. G-M1-MO-2')
print('CC1 = %.2f, CC2 = %.2f, CC3 = %.2f, Disparity = %.2f' %
      (pcs1G2_correlations[0], pcs1G2_correlations[1], pcs1G2_correlations[2], pcs1G2_disparity))

# G0914M1 vs G0915M1
cca.fit(pcsG1, pcsG2)
X_c, Y_c = cca.transform(pcsG1, pcsG2)
pcsG1G2_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
mtx1, mtx2, pcsG1G2_disparity = procrustes(pcsG1, pcsG2)
print('G-M1-MO-1 vs. G-M1-MO-2')
print('CC1 = %.2f, CC2 = %.2f, CC3 = %.2f, Disparity = %.2f' %
      (pcsG1G2_correlations[0], pcsG1G2_correlations[1], pcsG1G2_correlations[2], pcsG1G2_disparity))


#%% RNN
## set up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH_LOAD_models = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/models')
PATH_SAVE = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/results')

# conditiion
SP, SPText, nSP = ts.SP, ts.SPText, ts.nSP 
MD, MDText, nMD = ts.MD, ts.MDText, ts.nMD

# validation set
inputtype = 'Ca1'
nValidate = 400

np.random.seed(1125)
sp_array = SP[np.random.choice(range(len(SP)), (nValidate, 1))]
ti_array = np.random.rand(nValidate, 1) * np.pi * 2

test_cd = ts.StandardCondition(ti_array, sp_array)
inputs_here = test_cd.inputs(inputtype)
outputs_here = test_cd.outputs()

for i in range(100):
    load_name = 'rnn_Ca1_%d' % i
    print(load_name)

    input_size = 5
    hidden_size = 200
    output_size = 2

    init_state = (torch.zeros((1, hidden_size))).to('cpu')

    # load state dict
    load_path = PATH_LOAD_models + os.sep + load_name +  '.pth'
    model = rs.rRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))

    # get model results
    x_tensor = torch.from_numpy(np.array(inputs_here, dtype=np.float32))
    h, r, pred = model(x_tensor, init_state)

    pred_np = pred.cpu().detach().numpy()
    rate_np = r.cpu().detach().numpy()

    target_pos = test_cd.object_dict['target_pos']
    hand_pos = helper.getting_position_from_velocity(pred_np, test_cd)
    
    err_dist = []
    for itrial in range(hand_pos.shape[0]):
        temp = np.linalg.norm(
            hand_pos[itrial, test_cd.time_dict['time_total'][itrial] - 1, :]
            - target_pos[itrial, test_cd.time_dict['time_total'][itrial] - 1, :])
        err_dist.append(temp)
    err_dist = np.array(err_dist)
    error_mean, error_std = np.mean(err_dist), np.std(err_dist)
    correct_ratio = np.sum(err_dist < 0.02) / len(err_dist)

    target_enddir = np.array(
        [test_cd.object_dict['target_rad'][
            i, test_cd.time_dict['time_total'][i] - 1, 0] 
            for i in range(len(test_cd.object_dict['target_rad']))])

    # Conditional error distribution
    # corr_idx = np.arange(nValidate)
    corr_idx = np.argwhere(err_dist<0.02).flatten().tolist()
    corr_r_np = rate_np[corr_idx, :, :]
    corr_p_np = pred_np[corr_idx, :, :]
    corr_hp_np = hand_pos[corr_idx, :, :]

    nTrial, nTime, nNeuron = corr_r_np.shape

    error_df = pd.DataFrame(columns=['OriTrial', 
                                        'distance_error', 
                                        'direction_error'])
    for iTrial in corr_idx:
        endidx = test_cd.time_dict['time_total'][iTrial] - 1

        distance_error = err_dist[iTrial]

        touch_direction = np.arctan2(hand_pos[iTrial, endidx, 1],
                                        hand_pos[iTrial, endidx, 0])

        direction_error = touch_direction - target_enddir[iTrial]

        error_df = error_df._append([{'OriTrial': iTrial,
                                        'distance_error': distance_error,
                                        'direction_error': direction_error,
                                        'touch_direction': touch_direction,
                                        }], ignore_index=True)

    SPList, SPIdx, SPNum = np.unique(
        test_cd.condition_dict['target_speed'][corr_idx, 0],
        return_inverse=True, return_counts=True)
    MDIdx = helper.get_direction_split_index(error_df['touch_direction'], 8)
    MDList, MDNum = np.unique(MDIdx, return_counts=True)

    error_df['SP'] = test_cd.condition_dict['target_speed'][corr_idx, 0]
    error_df['SPIdx'] = SPIdx
    error_df['MDIdx'] = MDIdx

    # get firing rates
    rr = ac.getting_rate_slice(corr_r_np, test_cd, [-200, 120], 'mo')
    rr = rr[:300, :, :]
    rr = rr.reshape(300, 40, 8, 200)
    rr = rr.mean(axis=2)
    rr_ct = rr.reshape(rr.shape[0]*rr.shape[1], rr.shape[2])

    pca = PCA(n_components=30)
    r_pcs = pca.fit_transform(rr_ct)
    r_evr = pca.explained_variance_ratio_

    # C1022M1 vs RNN
    cca.fit(pcs1, r_pcs)
    X_c, Y_c = cca.transform(pcs1, r_pcs)
    pcs1r_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
    mtx1, mtx2, pcs1r_disparity = procrustes(pcs1, r_pcs)

    # G0914M1 vs RNN
    cca.fit(pcsG1, r_pcs)

    X_c, Y_c = cca.transform(pcsG1, r_pcs)
    pcsG1r_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
    mtx1, mtx2, pcsG1r_disparity = procrustes(pcsG1, r_pcs)

    # G0915M1 vs RNN
    cca.fit(pcsG2, r_pcs)
    X_c, Y_c = cca.transform(pcsG2, r_pcs)
    pcsG2r_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
    mtx1, mtx2, pcsG2r_disparity = procrustes(pcsG2, r_pcs)

    # shuffle across time
    rr_s1 = rr.copy() 
    for i in range(300):
        for j in range(200):
            aa = rr_s1[i, :, j]
            np.random.shuffle(aa)
            rr_s1[i, :, j] = aa
    rr_s1_t = rr_s1.reshape(rr.shape[0]*rr.shape[1], rr.shape[2])

    pca = PCA(n_components=30)
    rs_pcs = pca.fit_transform(rr_s1_t)
    rs_evr = pca.explained_variance_ratio_

    # C1022M1 vs RNN shuffle
    cca.fit(pcs1, rs_pcs)
    X_c, Y_c = cca.transform(pcs1, rs_pcs)
    pcs1rs_correlations = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])])
    mtx1, mtx2, pcs1rs_disparity = procrustes(pcs1, rs_pcs)


    #%% summarize

    similarty_df = pd.DataFrame(columns = ['CCA', 'Procrustes'])
  
    similarty_df.loc['C-M1-MO vs. RNN-MO', 'CCA'] = pcs1r_correlations
    similarty_df.loc['C-M1-MO vs. RNN-MO', 'Procrustes'] = pcs1r_disparity

    similarty_df.loc['G-M1-MO-1 vs. RNN-MO', 'CCA'] = pcsG1r_correlations
    similarty_df.loc['G-M1-MO-1 vs. RNN-MO', 'Procrustes'] = pcsG1r_disparity

    similarty_df.loc['G-M1-MO-2 MO vs. RNN-MO', 'CCA'] = pcsG2r_correlations
    similarty_df.loc['G-M1-MO-2 MO vs. RNN-MO', 'Procrustes'] = pcsG2r_disparity

    similarty_df.loc['C-M1-MO vs. RNN-MO-shuffle', 'CCA'] = pcs1rs_correlations
    similarty_df.loc['C-M1-MO vs. RNN-MO-shuffle', 'Procrustes'] = pcs1rs_disparity

    for i in range(4):
        for j in range(10):
            similarty_df.loc[similarty_df.index[i], 'CC%d'%j] = similarty_df.loc[similarty_df.index[i], 'CCA'].data[j]

    # print(similarty_df)
    similarty_df.to_csv(os.path.join(PATH_SAVE, '%s_CCA_PA_results.csv' % load_name), index=True)


#%%
CC0_list = []
CC1_list = []
CC2_list = []
PA_list = []

for i in range(100):
# i = 0
    load_name = 'rnn_Ca1_%d' % i
    load_path = PATH_SAVE + os.sep + load_name + os.sep
    
    similarty_df = pd.read_csv(os.path.join(PATH_SAVE, '%s_CCA_PA_results.csv' % load_name))
        
    CC0_list.append(similarty_df['CC0'].values)
    CC1_list.append(similarty_df['CC1'].values)
    CC2_list.append(similarty_df['CC2'].values)
    PA_list.append(similarty_df['Procrustes'].values)

pairs = similarty_df['Unnamed: 0'].values.tolist()

CC0_mean = np.array(CC0_list).mean(axis=0)
CC0_std = np.array(CC0_list).std(axis=0)
CC1_mean = np.array(CC1_list).mean(axis=0)
CC1_std = np.array(CC1_list).std(axis=0)
CC2_mean = np.array(CC2_list).mean(axis=0)
CC2_std = np.array(CC2_list).std(axis=0)
PA_mean = np.array(PA_list).mean(axis=0)
PA_std = np.array(PA_list).std(axis=0)

sdf = pd.DataFrame(columns=['CC0', 'CC1', 'CC2', 'PA'])
for ind, i in enumerate(pairs):
    sdf.loc[i, 'CC0'] = ['%.2f' % CC0_mean[ind], '%.2f' % CC0_std[ind]]
    sdf.loc[i, 'CC1'] = ['%.2f' % CC1_mean[ind], '%.2f' % CC1_std[ind]]
    sdf.loc[i, 'CC2'] = ['%.2f' % CC2_mean[ind], '%.2f' % CC2_std[ind]]
    sdf.loc[i, 'PA'] = ['%.2f' % PA_mean[ind], '%.2f' % PA_std[ind]]

print(sdf)