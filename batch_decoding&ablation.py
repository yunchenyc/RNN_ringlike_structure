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
import random
import pickle
import matplotlib.pyplot as plt
import helpers.plot_functions as plot
import helpers.helper_functions as helper
import settings.RNN_setting as rs
import settings.task_setting as ts
import analyses.acquisition as ac
import analyses.classification as cl
import analyses.geometry as ge
import analyses.states as st
from analyses.blocking import blocking

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


global device, PATH_SAVE, PATH_LOAD, plot_figures, save_variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH_SAVE = format(r'/AMAX/cuihe_lab/chenyun/code/CR_version_spyder/revision_results')
PATH_LOAD = format(r'/AMAX/cuihe_lab/chenyun/code/CR_version_spyder/models')

# %% Task setup
def run_analyses(i):

    #%% conditiion and color
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
    # outputs_here = test_cd.outputs()
    
    #%% set load name and save path 
    load_name = 'rnn_Ca1_%d' % i
    print(load_name)
    
    save_path = PATH_SAVE + os.sep + load_name + os.sep
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    print(save_path)
        

    # %% load RNN
    
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
    

    #%% get model results
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
    print(correct_ratio)
    
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
        
    
    #%% SVM decoding
    rate_to = ac.getting_rate_slice(corr_r_np, test_cd, (0, 300), 'to')
    rate_go = ac.getting_rate_slice(corr_r_np, test_cd, (-300, 300), 'go')
    rate_mo = ac.getting_rate_slice(corr_r_np, test_cd, (-300, 100), 'mo')

    pca = PCA(n_components=100)

    def svm_test(fr, t_max, cla_idx):
        cla_acc_ls = []    
        for j in range(10):
            acc_ls = []
            for i in np.arange(50, t_max, 50):
                x_slice = np.mean(fr[:, i-50:i+50, :], axis=1)
                x_slice = pca.fit_transform(x_slice)
                x_train, x_test, y_train, y_test = train_test_split(x_slice, cla_idx, test_size=0.1, random_state=j)

                svm = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr')
                svm.fit(x_train, y_train)
                y_pred = svm.predict(x_test)
                acc_ls.append(accuracy_score(y_test, y_pred))
            cla_acc_ls.append(acc_ls)
        
        return cla_acc_ls

    svm_sp_to = np.array(svm_test(rate_to, 300, SPIdx))
    svm_sp_go = np.array(svm_test(rate_go, 600, SPIdx))
    svm_sp_mo = np.array(svm_test(rate_mo, 400, SPIdx))

    svm_md_to = np.array(svm_test(rate_to, 300, MDIdx))
    svm_md_go = np.array(svm_test(rate_go, 600, MDIdx))
    svm_md_mo = np.array(svm_test(rate_mo, 400, MDIdx))

    plt.figure()
    plt.plot(np.arange(50, 300, 50), np.mean(svm_md_to, axis=0), 'k-')
    plt.plot(np.arange(50, 300, 50), np.mean(svm_sp_to, axis=0), 'b-')

    plt.plot(np.arange(350, 900, 50), np.mean(svm_md_go, axis=0), 'k-')
    plt.plot(np.arange(350, 900, 50), np.mean(svm_sp_go, axis=0), 'b-')

    plt.plot(np.arange(950, 1300, 50), np.mean(svm_md_mo, axis=0), 'k-')
    plt.plot(np.arange(950, 1300, 50), np.mean(svm_sp_mo, axis=0), 'b-')

    plt.xticks([0, 200, 400, 600, 800, 1000, 1200],
               labels=['TO', 200, -200, 'GO', 200, -200, 'MO'])

    plt.savefig(save_path + 'f20_decoding_step10ms.png', dpi=300)
    plt.show() 
    
    svm_results = {'sp': [svm_sp_to, svm_sp_go, svm_sp_mo],
                   'md': [svm_md_to, svm_md_go, svm_md_mo]}

            
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
    

    # %% Ablation

    period = 'MO'
    ablation_df = pd.DataFrame(columns=['error_distance', 'correct_ratio',
                                        'R2', 'within_mean', 'control_mean'])
    
    # cover
    for mod in ['S', 'G', 'A']:
        
        model_ia = rs.rRNN(model.input_size, 
                           model.hidden_size, 
                           model.output_size)
        model_ia.load_state_dict(model.state_dict())
        
        
        for ia in [i for i in dict_mol[period][mod]]:
            model_ia.state_dict()['cell.w_ih.weight'][ia, :] = 0
            model_ia.state_dict()['cell.w_hh.weight'][:, ia] = 0
            model_ia.state_dict()['cell.w_hh.weight'][ia, :] = 0            
            model_ia.state_dict()['cell.w_ho.weight'][:, ia] = 0
                            
        h1, r1, pred1 = model_ia(x_tensor, init_state)
        
        p1_np = pred1.detach().numpy()
        r1_np = r1.detach().numpy()
        target_pos = test_cd.object_dict['target_pos']
        hand_pos1 = helper.getting_position_from_velocity(p1_np, test_cd)
        
        
        ablation_dict, pcs1, evr1, SPIdx1, MDIdx1 = blocking(
            r1_np, p1_np, test_cd, SPText)
        ablation_df.loc['%s' % mod, :] = ablation_dict

        
        plot.f18_neural_states_perturbed(pcs1, evr1, SPIdx1, color_map_sp, 
                                            'SP_%s' % mod,
                                            save_path)
        plot.f18_neural_states_perturbed(pcs1, evr1, MDIdx1, color_map_md,
                                            'MD_%s' % mod,
                                            save_path)

        index = [1, 2, 3, 4, 5]
        plot.f19_hand_trajectory(target_pos[index, :, :], 
                                    hand_pos1[index, :, :], 
                                    test_cd.time_dict['time_total'][index], 
                                    index, '%s' % mod,
                                    save_path)    
    

    # only 
    only_ablation_df = pd.DataFrame(columns=['error_distance', 'correct_ratio',
                                        'R2', 'within_mean', 'control_mean'])
    for mod in ['S', 'G', 'A']:
        
        model_ia = rs.rRNN(model.input_size, 
                           model.hidden_size, 
                           model.output_size)
        model_ia.load_state_dict(model.state_dict())
        
        other = [i for i in ['S', 'G', 'A'] if i!=mod]
        for ia in [i for i in dict_mol[period][mod] if (i not in dict_mol[period][other[0]]) and (i not in dict_mol[period][other[1]])]:
            model_ia.state_dict()['cell.w_ih.weight'][ia, :] = 0
            model_ia.state_dict()['cell.w_hh.weight'][:, ia] = 0
            model_ia.state_dict()['cell.w_hh.weight'][ia, :] = 0            
            model_ia.state_dict()['cell.w_ho.weight'][:, ia] = 0
                            
        h1, r1, pred1 = model_ia(x_tensor, init_state)
        
        p1_np = pred1.detach().numpy()
        r1_np = r1.detach().numpy()
        target_pos = test_cd.object_dict['target_pos']
        hand_pos1 = helper.getting_position_from_velocity(p1_np, test_cd)
        
        
        only_ablation_dict, pcs1, evr1, SPIdx1, MDIdx1 = blocking(
            r1_np, p1_np, test_cd, SPText)
        only_ablation_df.loc['%s' % mod, :] = only_ablation_dict

        
        plot.f18_neural_states_perturbed(pcs1, evr1, SPIdx1, color_map_sp, 
                                            'SP_only%s' % mod,
                                            save_path)
        plot.f18_neural_states_perturbed(pcs1, evr1, MDIdx1, color_map_md,
                                            'MD_only%s' % mod,
                                            save_path)

        index = [1, 2, 3, 4, 5]
        plot.f19_hand_trajectory(target_pos[index, :, :], 
                                    hand_pos1[index, :, :], 
                                    test_cd.time_dict['time_total'][index], 
                                    index, 'only%s' % mod,
                                    save_path)
        

#%%
    
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
                'decoding': svm_results,
                'ablation': ablation_df,
                'ablation_only': only_ablation_df}
        
    pickle.dump(results, open('%s%s_results.pkl'%(save_path, load_name), 'wb'))
    
    # decoding_results = {'decoding': svm_results}
    # pickle.dump(decoding_results, open('%s%s_decoding_results.pkl'%(save_path, load_name), 'wb'))


#%%
for i in np.arange(100):
    load_name = 'rnn_Ca1_%d' % i
    training=np.load(PATH_LOAD + os.sep + load_name + '.npy', 
                     allow_pickle=True).item()
    loss_list=training['loss_list']
    if len(loss_list)<500:
        print(i)

    run_analyses(i)
    
    
