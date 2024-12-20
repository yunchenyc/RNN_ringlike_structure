# -*- coding: utf-8 -*-
"""
Created for Interception

@author: yc
"""
    
import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
# import random
import pickle
# import matplotlib.pyplot as plt
import helpers.plot_functions as plot
import helpers.helper_functions as helper
import settings.RNN_setting as rs
import settings.task_setting as ts
import analyses.acquisition as ac
import analyses.classification as cl
import analyses.geometry as ge
import analyses.states as st
from analyses.blocking import blocking


global device, PATH_SAVE, PATH_LOAD, plot_figures, save_variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH_SAVE = format(r'/AMAX/cuihe_lab/chenyun/code/CR_version_spyder/results')
PATH_LOAD = format(r'/AMAX/cuihe_lab/chenyun/code/CR_version_spyder/models')

plot_figures = 1
save_variables = 1


# %% Task setup
def run_analyses(i):

    # conditiion and color
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
    if plot_figures:
        
        plot.f1_inputs_outputs_overview(inputs_here, outputs_here, inputtype, 
                                        PATH_SAVE+os.sep)
        plot.f2_inputs(inputs_here, inputtype, test_cd, PATH_SAVE+os.sep)
        plot.f3_outputs(outputs_here, test_cd, PATH_SAVE+os.sep)


# %%
    load_name = 'rnn_Ca1_%d' % i
    print(load_name)
    
    save_path = PATH_SAVE + os.sep + load_name + os.sep
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
        
    # %% Load RNN
    
    input_size = 5
    hidden_size = 200
    output_size = 2
    
    # init_state = (torch.rand((1, hidden_size))*0.3).to(device) #.cuda()
    init_state = (torch.zeros((1, hidden_size))).to('cpu')
    
    # load state dict
    # load_name = 'rnn_Ca1_0'
    load_path = PATH_LOAD + os.sep + load_name +  '.pth'
    model = rs.rRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    
    # get model results
    x_tensor = torch.from_numpy(np.array(inputs_here, dtype=np.float32))
    h, r, pred = model(x_tensor, init_state)
    
    pred_np = pred.cpu().detach().numpy()
    rate_np = r.cpu().detach().numpy()
    
    target_pos = test_cd.object_dict['target_pos']
    hand_pos = helper.getting_position_from_velocity(pred_np, test_cd)
    # hand_pos = helper.getting_position_from_velocity(outputs_here, test_cd)
    
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
    
    _, sp_bhv_p = stats.kruskal(
        *list(
            error_df['distance_error'].groupby(
                error_df['SPIdx']).apply(np.array).values))
    
    _, md_bhv_p = stats.kruskal(
        *list(
            error_df['distance_error'].groupby(
                error_df['MDIdx']).apply(np.array).values))
    
    
    if plot_figures:
        
        # plot 
        plot.f4_output_compared_with_target(pred_np, outputs_here, save_path)
        
        index = [1, 2, 3, 4, 5]
        plot.f5_hand_trajectory(target_pos[index, :, :], hand_pos[index, :, :], 
                                test_cd.time_dict['time_total'][index], index,
                                save_path)
        
        plot.f6_groupby_distance_error(error_df, mypalsp, mypalmd, SPText, MDText,
                                       save_path)    
        plot.f7_condition_distribution(SPNum, MDNum, save_path)
    
    
    # %% Classification by modulation
    co_idx = np.argwhere(SPList == 0)[0, 0]
    
    # [TO+200, TO+400]
    rate_to, nidx_to = \
        ac.getting_nonzero_rate_slice(corr_r_np, (SPIdx, MDIdx), test_cd,
                                      (200, 400), 'to')
    
    # gain, addition, shift
    gnp, anp, snp = cl.getting_modulation_sig(rate_to, 20, 'all', 
                                              (SPIdx, MDIdx), co_idx)
    to_modul_np = np.zeros((rate_to.shape[2], 3))
    for i, t in enumerate([snp, gnp, anp]):
        for iNZ in range(rate_to.shape[2]):
            if (t[iNZ] > -1) & (t[iNZ] < 0.05):
                to_modul_np[iNZ, i] = 1
            
            
    # [GO-100, GO+100]
    rate_go, nidx_go = \
        ac.getting_nonzero_rate_slice(corr_r_np, (SPIdx, MDIdx), test_cd,
                                      (-100, 100), 'go')
    
    # gain, addition, shift
    gnp, anp, snp = cl.getting_modulation_sig(rate_go, 20, 'all', 
                                              (SPIdx, MDIdx), co_idx)
    
    go_modul_np = np.zeros((rate_go.shape[2], 3))
    for i, t in enumerate([snp, gnp, anp]):
        for iNZ in range(rate_go.shape[2]):
            if (t[iNZ] > -1) & (t[iNZ] < 0.05):
                go_modul_np[iNZ, i] = 1
     
    
    # [MO-100, MO+100]    
    rate_mo, nidx_mo = \
        ac.getting_nonzero_rate_slice(corr_r_np, (SPIdx, MDIdx), test_cd,
                                      (-100, 100), 'mo')
    
    # gain, addition, shift
    gnp, anp, snp = cl.getting_modulation_sig(rate_mo, 20, 'all', 
                                              (SPIdx, MDIdx), co_idx)
    
    mo_modul_np = np.zeros((rate_mo.shape[2], 3))
    for i, t in enumerate([snp, gnp, anp]):
        for iNZ in range(rate_mo.shape[2]):
            if (t[iNZ] > -1) & (t[iNZ] < 0.05):
                mo_modul_np[iNZ, i] = 1
    
    print(to_modul_np.shape)
    print(go_modul_np.shape)
    print(mo_modul_np.shape)
    
    print(np.sum(to_modul_np, axis=0))
    print(np.sum(go_modul_np, axis=0))
    print(np.sum(mo_modul_np, axis=0))
    
    dict_mol = {}
    for period in ['TO', 'GO', 'MO']:
        dict_mol[period] = {'S': eval('np.asarray(nidx_%s)[%s_modul_np[:, 0]==1]'
                                      % (period.lower(), period.lower())),
                            'G': eval('np.asarray(nidx_%s)[%s_modul_np[:, 1]==1]'
                                      % (period.lower(), period.lower())),
                            'A': eval('np.asarray(nidx_%s)[%s_modul_np[:, 2]==1]'
                                      % (period.lower(), period.lower()))}
        dict_mol[period]['N'] = eval(
            'np.asarray(nidx_%s)[np.sum(%s_modul_np, axis=1)==0]'
            % (period.lower(), period.lower()))
    
    # print(dict_mol['TO']['S'])
    
    all_n_mol = np.zeros((nNeuron, 9))
    for ineuron in range(nNeuron):
        all_n_mol[ineuron, 0] = int(ineuron in dict_mol['TO']['S'])
        all_n_mol[ineuron, 1] = int(ineuron in dict_mol['TO']['G'])
        all_n_mol[ineuron, 2] = int(ineuron in dict_mol['TO']['A'])
        all_n_mol[ineuron, 3] = int(ineuron in dict_mol['GO']['S'])
        all_n_mol[ineuron, 4] = int(ineuron in dict_mol['GO']['G'])
        all_n_mol[ineuron, 5] = int(ineuron in dict_mol['GO']['A'])
        all_n_mol[ineuron, 6] = int(ineuron in dict_mol['MO']['S'])
        all_n_mol[ineuron, 7] = int(ineuron in dict_mol['MO']['G'])
        all_n_mol[ineuron, 8] = int(ineuron in dict_mol['MO']['A'])
    
    
    if plot_figures:
        
        plot.f8_modulation_venn(to_modul_np, go_modul_np, mo_modul_np, save_path)
                
        for inn in nidx_mo: #[3, 9, 11]:
            r_np = np.mean(rate_mo[:, :, nidx_mo.index(inn)], axis=1)
            plot.f9_neuronal_modulation_radar(r_np, inn, nSP, nMD, save_path)
        
        
            rate_plot = ac.getting_rate_slice(corr_r_np, test_cd, (-300, 150), 
                                          'mo', nidx_mo)
        
            plot.f10_neuronal_conditional_rates(rate_plot[:, :, nidx_mo.index(inn)], 
                                                inn, SPIdx, MDIdx, save_path)
    
    
    
    # %% Neural states
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
    
    
    if plot_figures:
        # plot.plot_fitted_ellispes_(pcs, SPIdx)
    
        # 3d
        plot.f11_neural_states_3d(pcs, evr, SPIdx, color_map_sp, 'SP', save_path)
        plot.f11_neural_states_3d(pcs, evr, MDIdx, color_map_md, 'MD', save_path)
        
        # 2d x 3
        plot.f12_neural_states_2d(pcs, evr, SPIdx, color_map_sp, 'SP', save_path)
        plot.f12_neural_states_2d(pcs, evr, MDIdx, color_map_md, 'MD', save_path)
            
        
        # fitted plane of single SP
        for i in np.unique(SPIdx):
            plot.f13_states_fitted_plane((pcs[SPIdx==i, 0], 
                                          pcs[SPIdx==i, 1], 
                                          pcs[SPIdx==i, 2]), 
                                         geometry_df.loc[SPText[i], 'plane'][0], 
                                         SPText[i], save_path)
        
        # fitted planes of all SP
        plot.f13_states_fitted_plane(
            [(pcs[SPIdx==i, 0], pcs[SPIdx==i, 1], pcs[SPIdx==i, 2]) 
             for i in np.unique(SPIdx)],
            [geometry_df.loc[SPText[i], 'plane'][0] for i in np.unique(SPIdx)], 
            'all', save_path)
        
        # fitted plane of single SP
        for i in np.unique(SPIdx):
            plot.f14_states_fitted_ellipses((pcs[SPIdx==i, 0], 
                                             pcs[SPIdx==i, 1], 
                                             pcs[SPIdx==i, 2]),
                                            evr,
                                            geometry_df.loc[SPText[i], 'ellipses'][0],
                                            SPIdx[SPIdx==i],
                                            color_map_sp,
                                            SPText[i],
                                            save_path)
        
        # fitted planes of all SP
        plot.f14_states_fitted_ellipses(
            [(pcs[SPIdx==i, 0], pcs[SPIdx==i, 1], pcs[SPIdx==i, 2]) 
             for i in np.unique(SPIdx)],
            evr,
            [geometry_df.loc[SPText[i], 'ellipses'][0] for i in np.unique(SPIdx)], 
            SPIdx, color_map_sp, 'all', save_path)
        
       
        # angles
        plot.f15_angle_between_planes(SPList / np.pi * 180, theta_pp_np, 
                                      aa, bb, lf, 
                                      color_map_sp(range(nSP)), save_path)
        
        plot.f16_angle_elevation_azimuth(SPList, theta_z_np, theta_xy_np, 
                                         co_idx, color_map_sp(np.arange(nSP)),
                                         save_path)
            
    
    # %% Connection distribution
    
    w_ih = model.state_dict()['cell.w_ih.weight'].numpy()  # (nHidden x nInput)
    w_hh = model.state_dict()['cell.w_hh.weight'].numpy()
    w_ho = model.state_dict()['cell.w_ho.weight'].numpy()
    
    connec_mol = np.zeros((3, 3))
    for ii, itype in enumerate(['S', 'G', 'A']):
        for jj, jtype in enumerate(['S', 'G', 'A']):
            temp = []
            for ia in dict_mol['MO'][itype]:
                temp.append(np.mean(w_hh[dict_mol['MO'][jtype], ia]))
            connec_mol[ii, jj] = np.mean(np.asarray(temp))
    
    
    if plot_figures:
    
        plot.f17_connectivity_between_modulations(w_hh, dict_mol, 
                                                  ['S', 'G', 'A'], save_path)
    
        
    # %% Adjusting connections
    period = 'GO'
    blocking_df = pd.DataFrame(columns=['error_distance', 'correct_ratio',
                                        'R2', 'within_mean', 'control_mean'])
    for from_mod in ['S', 'G', 'A']:
        for to_mod in ['S', 'G', 'A']:
            
            model_ia = rs.rRNN(model.input_size, 
                               model.hidden_size, 
                               model.output_size)
            model_ia.load_state_dict(model.state_dict())
           
            
            for ia in [i for i in dict_mol[period][from_mod]]:
                model_ia.state_dict()[
                    'cell.w_hh.weight'][
                        [i for i in dict_mol[period][to_mod]], ia] = 0
                                
            h1, r1, pred1 = model_ia(x_tensor, init_state)
            
            p1_np = pred1.detach().numpy()
            r1_np = r1.detach().numpy()
            target_pos = test_cd.object_dict['target_pos']
            hand_pos1 = helper.getting_position_from_velocity(p1_np, test_cd)
          
            
            blocking_dict, pcs1, evr1, SPIdx1, MDIdx1 = blocking(
                r1_np, p1_np, test_cd, SPText)
            blocking_df.loc['%s to %s' % (from_mod, to_mod), :] = blocking_dict
    
            if plot_figures:
              
                plot.f18_neural_states_perturbed(pcs1, evr1, SPIdx1, color_map_sp, 
                                                 'SP_%sto%s' % (from_mod, to_mod),
                                                 save_path)
                plot.f18_neural_states_perturbed(pcs1, evr1, MDIdx1, color_map_md,
                                                 'MD_%sto%s' % (from_mod, to_mod),
                                                 save_path)

                index = [1, 2, 3, 4, 5]
                plot.f19_hand_trajectory(target_pos[index, :, :], 
                                         hand_pos1[index, :, :], 
                                         test_cd.time_dict['time_total'][index], 
                                         index, '%sto%s' % (from_mod, to_mod),
                                         save_path)


#%%
    if save_variables:
        results = {'test': test_cd,
                   'performance': {'rate': rate_np, 'pred': pred_np,
                                   'hand_pos': hand_pos, 
                                   'error_dist': err_dist,
                                   'corr_idx': corr_idx,
                                   'corr_error_df': error_df},
                   'modulation': {'to_modul_np': to_modul_np,
                                  'go_modul_np': go_modul_np,
                                  'mo_modul_np': mo_modul_np,
                                  'dict_mol': dict_mol,
                                  'all_n_mol': all_n_mol},
                   'geometry': {'pcs': pcs, 'evr': evr,
                                'geometry_df': geometry_df,
                                'theta_pp_np': theta_pp_np,
                                'theta_xy_np': theta_xy_np,
                                'theta_z_np': theta_z_np},
                   'connection': connec_mol,
                   'blocing': blocking_df}
        
        pickle.dump(results, open('%s%s_results.pkl'%(save_path, load_name), 'wb'))
        


#%%
for i in range(100):
    load_name = 'rnn_Ca1_%d' % i
    training=np.load(PATH_LOAD + os.sep + load_name + '.npy', 
                     allow_pickle=True).item()
    loss_list=training['loss_list']
    if len(loss_list)<500:
        print(i)
    
    
for i in range(100):
# i = 28

    run_analyses(i)
    # try:
    #     run_analyses(i)
    # except KeyError : 
    #     print(i)
    #     pass
    #     continue


