#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import helpers.plot_functions as plot
import settings.task_setting as ts
import settings.RNN_setting as rs
import helpers.helper_functions as helper
import analyses.acquisition as ac
import analyses.geometry as ge
import analyses.states as st
import torch
from scipy import optimize

PATH_LOAD_models = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/models')
PATH_LOAD_results = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/results')
PATH_SAVE = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/outputs')


#%% load individual results 
results_list = []
for i in range(100):

    load_name = 'rnn_Ca1_%d' % i

    with open(os.path.join(PATH_LOAD_results, '%s_results.pkl' % load_name), 'rb') as f:
        results = pickle.load(f)
        results_list.append(results)


#%% Example input and output
### Figure 5A ###
SP, SPText, nSP = ts.SP, ts.SPText, ts.nSP 
color_map_sp, mypalsp = ts.color_map_sp, ts.mypalsp

MD, MDText, nMD = ts.MD, ts.MDText, ts.nMD
color_map_md, mypalmd = ts.color_map_md, ts.mypalmd

# new validation set
np.random.seed(4616)

inputtype = 'Ca1'
nValidate = 500
sp_array = SP[np.random.choice(range(len(SP)), (nValidate, 1))]
ti_array = np.random.rand(nValidate, 1) * np.pi * 2

test_cd = ts.StandardCondition(ti_array, sp_array)
inputs_here = test_cd.inputs(inputtype)
outputs_here = test_cd.outputs()

# timeline    
plot.f2_inputs(inputs_here[[2], :, :], inputtype, test_cd, PATH_SAVE+os.sep)
plot.f3_outputs(outputs_here[[2], :, :], test_cd, PATH_SAVE+os.sep)


#%% Example neuron
### Figure 5B ###             
# Load RNN

input_size = 5
hidden_size = 200
output_size = 2

# init_state = (torch.rand((1, hidden_size))*0.3).to(device) #.cuda()
init_state = (torch.zeros((1, hidden_size))).to('cpu')

# load state dict
load_name = 'rnn_Ca1_97'
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

_, sp_bhv_p = stats.kruskal(
    *list(
        error_df['distance_error'].groupby(
            error_df['SPIdx']).apply(np.array).values))

_, md_bhv_p = stats.kruskal(
    *list(
        error_df['distance_error'].groupby(
            error_df['MDIdx']).apply(np.array).values)) 

# [MO-100, MO+100]    
rate_mo, nidx_mo = \
    ac.getting_nonzero_rate_slice(corr_r_np, (SPIdx, MDIdx), test_cd,
                                    (-100, 100), 'mo')
            
for inn in [12, 195, 197]:
    r_np = np.mean(rate_mo[:, :, nidx_mo.index(inn)], axis=1)

    rate_plot = ac.getting_rate_slice(corr_r_np, test_cd, (-300, 150), 
                                    'mo', nidx_mo)

    plot.f10_neuronal_conditional_rates(rate_plot[:, :, nidx_mo.index(inn)], 
                                        inn, SPIdx, MDIdx, PATH_SAVE+os.sep)



#%% Nodes Geometry
### Figure 5C ###
pcs, evr = st.getting_neural_states_withPCA(
    corr_r_np, -100, 0, 'mo', 2, test_cd)

geometry_df = pd.DataFrame(columns=['pcs_points', 'plane', 'ellipses', 'e-r2'])
for isp in np.unique(SPIdx):
    pcs_points = (pcs[SPIdx==isp, 0], pcs[SPIdx==isp, 1], pcs[SPIdx==isp, 2])
    
    plane_points, pp_points, pparams = \
        ge.getting_fitted_ellipses(pcs_points, 'plane')

    el_points, pel_points, elparams = \
        ge.getting_fitted_ellipses(pcs_points, 'ellipses')

    r2 = ge.r_square(np.array(pel_points), np.array(pcs_points))
    
    geometry_df.loc[SPText[isp], 'pcs_points'] = pcs_points
    geometry_df.loc[SPText[isp], 'plane'] = (plane_points, pp_points, pparams)
    geometry_df.loc[SPText[isp], 'ellipses'] = (el_points, pel_points, elparams)
    geometry_df.loc[SPText[isp], 'e-r2'] = r2
    
theta_pp_np, theta_xy_np, theta_z_np = \
    ge.getting_angle_between_planes(pcs, SPIdx, SPList)

theta_pp_np[np.isnan(theta_pp_np)] = 0

def lf(x, a, b):
    return a * x + b

# theta_pp_np[SPList == 0] = 0
aa, bb = optimize.curve_fit(lf, SPList / np.pi * 180, theta_pp_np)[0]

r_square_here = \
    1 - np.sum((lf(SPList / np.pi * 180, aa, bb) - theta_pp_np) ** 2) / np.sum(
    (theta_pp_np - np.mean(theta_pp_np)) ** 2)
print(r_square_here)

# 3d
plot.f11_neural_states_3d(pcs, evr, SPIdx, color_map_sp, 'SP', PATH_SAVE+os.sep)
plot.f11_neural_states_3d(pcs, evr, MDIdx, color_map_md, 'MD', PATH_SAVE+os.sep)
    


#%% Titling angle
### Figure 5D ###
theta_pp_list = [results_list[i]['geometry']['theta_pp_np']
                  for i in range(100)]
theta_pp_np = np.array(theta_pp_list)
theta_pp_np[theta_pp_np[:, 4]<-20, :] = - theta_pp_np[theta_pp_np[:, 4]<-20, :]

def lf(x, a, b):
    return a * x + b

# theta_pp_np[SPList == 0] = 0
aa, bb = optimize.curve_fit(lf, 
                            (np.tile(SPList[np.newaxis, :], (100, 1))/np.pi*180).flatten(), 
                            theta_pp_np.flatten())[0]
print('y = %.2fx + %.2f' % (aa, bb))

plt.figure(figsize=(4,3), dpi=300)
plt.plot(np.linspace(-360, 360, 100),
         lf(np.linspace(-360,360,100), aa, bb), 'k--')
plt.scatter(np.tile(np.array([[-240, -120, 0, 120, 240]]), (100, 1)), theta_pp_np)
plt.xticks([-240, -120, 0, 120, 240])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(PATH_SAVE, 'figure_5D.png'), dpi=300)
plt.show()

yfit = lf((np.tile(SPList[np.newaxis, :], (100, 1))/np.pi*180).flatten(), aa, bb)
yactual = theta_pp_np.flatten()
print('R-square = %.2f' % (1-np.sum((yfit-yactual)**2)/np.sum((yactual-np.mean(yactual))**2)))


# %% Connection
### Figure 5E ###
rlist = []
for i in range(100):
    input_size = 5
    hidden_size = 200
    output_size = 2
    
    load_name = 'rnn_Ca1_%d' % i
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
for i in range(100):
    for ii, itype in enumerate(['S', 'G', 'A']):
        for jj, jtype in enumerate(['S', 'G', 'A']):
             connection_df.loc[i, '%sto%s' % (itype, jtype)] = rlist[i][ii, jj]
print(connection_df)

plt.figure(figsize=(6,4), dpi=300)
plt.boxplot(connection_df, notch=True)
plt.xticks(ticks=range(1, 10), labels=['StoS', 'StoG', 'StoA',
                                       'GtoS', 'GtoG', 'GtoA',
                                       'AtoS', 'AtoG', 'AtoA'])
plt.savefig(os.path.join(PATH_SAVE, 'figure_5E.png'), dpi=300)
plt.show()

stat, p = stats.kruskal(*[connection_df.loc[:, i] 
                         for i in connection_df.columns])

import scikit_posthocs as sp
p_dunn = sp.posthoc_dunn([connection_df.loc[:, i] 
                         for i in connection_df.columns], p_adjust='holm')
print(p)
