#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
import settings.RNN_setting as rs
import analyses.geometry as ge
import torch
from scipy import optimize


PATH_LOAD_models = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/alternative_models')
PATH_LOAD_results = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/results')
PATH_SAVE = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/outputs')


#%% load individual results 
results_list = []
for i in range(10):

    # load_name = 'rnn_C1_%d' % i  #GM
    # load_name = 'rnn_a1_%d' % i  #GT
    load_name = 'sparse_rnn_Ca1_%d' % i  #sINIT

    with open(os.path.join(PATH_LOAD_results, '%s_results.pkl' % load_name), 'rb') as f:
        results = pickle.load(f)
        results_list.append(results)


#%% distance error
error_dist_list = [results_list[i]['performance']['error_dist']
                   for i in range(10)]
error_dist_np = np.array(error_dist_list)
error_mean = np.mean(error_dist_np, axis=1)
error_std = np.std(error_dist_np, axis=1)

print('error_mean = %.4f' % np.mean(error_dist_np.flatten()))
print('error_std = %.4f' % np.std(error_dist_np.flatten()))

correct_ratio_list = [len(results_list[i]['performance']['corr_idx'])/500
                      for i in range(10)]
correct_ratio_mean = np.mean(correct_ratio_list)
correct_ratio_std = np.std(correct_ratio_list)
print('correct ratio = %.2f +/- %.2f' % (correct_ratio_mean, correct_ratio_std))


#%% Modulation
mo_modul_list = [results_list[i]['modulation']['mo_modul_np']
                 for i in range(10)]
mo_modul_df = pd.DataFrame(columns=['G', 'S', 'A', 'G+S', 'S+A', 'G+A',
                                    'G+S+A', 'no modul / ac', 'no ac / all'])
# S, G, A in mo_modul_np
for i in range(10):
    mo_modul_df.loc[i, 'G'] = np.sum(mo_modul_list[i][:, 1]==1)/mo_modul_list[i].shape[0] #200
    mo_modul_df.loc[i, 'S'] = np.sum(mo_modul_list[i][:, 0]==1)/mo_modul_list[i].shape[0] #200
    mo_modul_df.loc[i, 'A'] = np.sum(mo_modul_list[i][:, 2]==1)/mo_modul_list[i].shape[0] #200
    
    mo_modul_df.loc[i, 'G+S'] = \
    np.sum((mo_modul_list[i][:, 1]==1)&(mo_modul_list[i][:, 0]==1))/mo_modul_list[i].shape[0]
    
    mo_modul_df.loc[i, 'S+A'] = \
    np.sum((mo_modul_list[i][:, 0]==1)&(mo_modul_list[i][:, 2]==1))/mo_modul_list[i].shape[0]
    
    mo_modul_df.loc[i, 'G+A'] = \
    np.sum((mo_modul_list[i][:, 1]==1)&(mo_modul_list[i][:, 2]==1))/mo_modul_list[i].shape[0]
    
    mo_modul_df.loc[i, 'G+S+A'] = \
    np.sum((mo_modul_list[i][:, 1]==1)&(mo_modul_list[i][:, 0]==1)
           &(mo_modul_list[i][:, 2]==1))/mo_modul_list[i].shape[0]
    
    mo_modul_df.loc[i, 'no modul / ac'] = \
    np.sum((mo_modul_list[i][:, 1]==0)&(mo_modul_list[i][:, 0]==0)
           &(mo_modul_list[i][:, 2]==0))/mo_modul_list[i].shape[0]
    
    mo_modul_df.loc[i, 'no ac / all'] = (200-mo_modul_list[i].shape[0])/200
    
print(mo_modul_df)
mo_modul_summary = pd.DataFrame(columns=['G', 'S', 'A', 'G+S', 'S+A', 'G+A',
                                         'G+S+A', 'no modul / ac', 'no ac / all'])
for c in mo_modul_summary.columns:
    mo_modul_summary.loc['mean', c] = '%.1f' % (np.mean(mo_modul_df[c])*100)
    mo_modul_summary.loc['std', c] = '%.1f' % (np.std(mo_modul_df[c])*100)
print(mo_modul_summary)


#%% PC explained variance
evr1, evr2, evr3 = [], [], []
for i in range(10):
    evr1.append(results_list[i]['geometry']['evr'][0])
    evr2.append(results_list[i]['geometry']['evr'][1])
    evr3.append(results_list[i]['geometry']['evr'][2])
print('PC1 exp. var. = %.1f ± %.1f' % (np.mean(evr1)*100, np.std(evr1)*100))
print('PC2 exp. var. = %.1f ± %.1f' % (np.mean(evr2)*100, np.std(evr2)*100))
print('PC3 exp. var. = %.1f ± %.1f' % (np.mean(evr3)*100, np.std(evr3)*100))


#%% Geometry

# R2 of fitting ellipses
def r_square(yfit, yactual):
    # yfit and yactual have same shape as (3 x nTrial)
    sse = np.sum(np.sum((yactual - yfit) ** 2, axis=1))
    sst = np.sum(np.sum((yactual - np.tile(np.mean(yactual, axis=1)[:, np.newaxis],
                                    (1, yactual.shape[1]))) ** 2, axis=1))

    return 1 - sse / sst

r100list = []
for i in range(10):
    cd = results_list[i]['test']
    corr_idx = results_list[i]['performance']['corr_idx']
    SPList, SPIdx, SPNum = np.unique(
        cd.condition_dict['target_speed'][corr_idx, 0],
        return_inverse=True, return_counts=True)
    rlist = []
    for isp, sp in enumerate(['-240', '-120', '0', '120', '240']):
        p_ori = results_list[i]['geometry']['pcs'][SPIdx==isp, :]
        pcs_points = (p_ori[:, 0], p_ori[:, 1], p_ori[:, 2])
        
        _, p_fit, _ = ge.getting_fitted_ellipses(pcs_points, 'ellipses')
        
        p_fit = np.asarray(p_fit)
        r2 = r_square(p_fit, p_ori.T)
        
        rlist.append(r2)
    # print(i, r2.mean())
    r100list.append(rlist)
r100np = np.array(r100list)
r100mean = np.mean(r100np.flatten())
r100std = np.std(r100np.flatten())
print('R-square of fitting ellipses =  %.4f ± %.4f' % (r100mean, r100std))


# R2 of fitting tilting angle (dim)
fitting_r_list = []
for i in range(10):
    temp = results_list[i]['geometry']['geometry_df']['e-r2'].values
    fitting_r_list.append(
        np.concatenate(temp).reshape((len(temp), len(temp[0]))))
                  
fitting_r_np = np.array(fitting_r_list)
fitting_r_mean_5sp = np.mean(fitting_r_np, axis=0) 
fitting_r_std_5sp = np.std(fitting_r_np, axis=0)
# print(fitting_r_mean_5sp)
# print(fitting_r_std_5sp)

fitting_r_mean = np.mean(fitting_r_np.reshape(
    (fitting_r_np.shape[0]*fitting_r_np.shape[1], 3)), axis=0)
fitting_r_std = np.std(fitting_r_np.reshape(
    (fitting_r_np.shape[0]*fitting_r_np.shape[1], 3)), axis=0)
print('R-square of fitting ellipses Dim 1 = %.4f ± %.4f' % (fitting_r_mean[0], fitting_r_std[0]))
print('R-square of fitting ellipses Dim 2 = %.4f ± %.4f' % (fitting_r_mean[1], fitting_r_std[1]))
print('R-square of fitting ellipses Dim 3 = %.4f ± %.4f' % (fitting_r_mean[2], fitting_r_std[2]))


# R2 of fitting tilting angle
theta_pp_list = [results_list[i]['geometry']['theta_pp_np']
                  for i in range(10)]
theta_pp_np = np.array(theta_pp_list)
theta_pp_np[theta_pp_np[:, 4]<-20, :] = - theta_pp_np[theta_pp_np[:, 4]<-20, :]

def lf(x, a, b):
    return a * x + b

# theta_pp_np[SPList == 0] = 0
aa, bb = optimize.curve_fit(lf, 
                            (np.tile(SPList[np.newaxis, :], (10, 1))/np.pi*180).flatten(), 
                            theta_pp_np.flatten())[0]

yfit = lf((np.tile(SPList[np.newaxis, :], (10, 1))/np.pi*180).flatten(), aa, bb)
yactual = theta_pp_np.flatten()
print('R-square of fitting tilting angle = %.4f' % (1-np.sum((yfit-yactual)**2)/np.sum((yactual-np.mean(yactual))**2)))


# %% Connection
rlist = []
for i in range(10):
    input_size = 5
    hidden_size = 200
    output_size = 2
    
    load_name = 'rnn_C1_%d' % i  #GM
    load_name = 'rnn_a1_%d' % i  #GT
    load_name = 'sparse_rnn_Ca1_%d' % i  #sINIT
    load_path = PATH_LOAD_models + os.sep + load_name +  '.pth'
    model = rs.rRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))     
    
    w_hh = model.state_dict()['cell.w_hh.weight'].numpy()
    
    dict_mol = results_list[i]['modulation']['dict_mol']['MO']
    connec_mol = np.zeros((3, 3))
    for ii, itype in enumerate(['S', 'G', 'A']):
       for jj, jtype in enumerate(['S', 'G', 'A']):  
            temp = []
            for ia in dict_mol[itype]:
                temp.append(np.mean(abs(w_hh[dict_mol[jtype], ia])))
            connec_mol[ii, jj] = np.mean(np.asarray(temp))
         
    rlist.append(connec_mol)
         
connection_df = pd.DataFrame(columns=['StoS', 'StoG', 'StoA',
                                      'GtoS', 'GtoG', 'GtoA',
                                      'AtoS', 'AtoG', 'AtoA'])
for i in range(10):
    for ii, itype in enumerate(['S', 'G', 'A']):
        for jj, jtype in enumerate(['S', 'G', 'A']):
             connection_df.loc[i, '%sto%s' % (itype, jtype)] = rlist[i][ii, jj]
print(connection_df)

connection_df = connection_df.dropna(how="all")
print(connection_df)

stat, p = stats.kruskal(*[connection_df.loc[:, i] 
                         for i in connection_df.columns])

import scikit_posthocs as sp
p_dunn = sp.posthoc_dunn([connection_df.loc[:, i] 
                         for i in connection_df.columns], p_adjust='holm')
print(p)