
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
import torch

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

results_list2 = []
for i in range(100):

    load_name = 'rnn_Ca1_%d' % i

    with open(os.path.join(PATH_LOAD_results, '%s_results2.pkl' % load_name), 'rb') as f:
        results = pickle.load(f)
        results_list2.append(results)


#%% Intact models
# distance error
error_dist_list = [results_list[i]['performance']['error_dist']
                   for i in range(100)]
error_dist_np = np.array(error_dist_list)
error_mean = np.mean(error_dist_np, axis=1)
error_std = np.std(error_dist_np, axis=1)

print('error_mean = %.4f' % np.mean(error_dist_np.flatten()))
print('error_std = %.4f' % np.std(error_dist_np.flatten()))

correct_ratio_list = [len(results_list[i]['performance']['corr_idx'])/500
                      for i in range(100)]
correct_ratio_mean = np.mean(correct_ratio_list)
correct_ratio_std = np.std(correct_ratio_list)
print('correct ratio = %.2f +/- %.2f' % (correct_ratio_mean, correct_ratio_std))

# fitting goodness
def r_square(yfit, yactual):
    # yfit and yactual have same shape as (3 x nTrial)
    sse = np.sum(np.sum((yactual - yfit) ** 2, axis=1))
    sst = np.sum(np.sum((yactual - np.tile(np.mean(yactual, axis=1)[:, np.newaxis],
                                    (1, yactual.shape[1]))) ** 2, axis=1))

    return 1 - sse / sst

r100list = []
for i in range(100):
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


#%% Ablation
blocking_df = pd.DataFrame(columns=['error_mean', 'error_std', 'R2', 
                                    'R2-dim3_mean', 'R2-dim3_std'])

for idx in results_list2[0]['ablation'].index:
    templist1 = []
    templist2 = []
    templist3 = []
    templist4 = []
    templist5 = []
    for i in range(100):
        templist1.append(results_list2[i]['ablation'].loc[idx, 'error_distance'][0])
        templist2.append(results_list2[i]['ablation'].loc[idx, 'error_distance'][1])
        temp = results_list2[i]['ablation'].loc[idx, 'R2']
       
        templist3.append(temp)
        templist4.append(np.mean(temp, axis=0))
        templist5.append(np.std(temp, axis=0))
    blocking_df.loc[idx, 'error_mean'] = templist1
    blocking_df.loc[idx, 'error_std'] = templist2
    blocking_df.loc[idx, 'R2'] = templist3
    blocking_df.loc[idx, 'R2-dim3_mean'] = templist4
    blocking_df.loc[idx, 'R2-dim3_std'] = templist5
        
# print(blocking_df)

blocking_df_summary = pd.DataFrame(columns=['error_mean_mean', 'error_mean_std'])
for idx in blocking_df.index:
    blocking_df_summary.loc[idx, 'error_mean_mean'] = \
        '%.4f' % np.mean(np.array(blocking_df.loc[idx, 'error_mean']))
    blocking_df_summary.loc[idx, 'error_mean_std'] = \
        '%.4f' % np.std(np.array(blocking_df.loc[idx, 'error_mean']))
    blocking_df_summary.loc[idx, 'R2_mean'] = \
        '%.4f' % np.mean(np.array(blocking_df.loc[idx, 'R2-dim3_mean']), axis=0)
    blocking_df_summary.loc[idx, 'R2_std'] = \
        '%.4f' % np.std(np.array(blocking_df.loc[idx, 'R2-dim3_mean']), axis=0)
  
print(blocking_df_summary)

## only
blocking_df = pd.DataFrame(columns=['error_mean', 'error_std', 'R2', 
                                    'R2-dim3_mean', 'R2-dim3_std'])
for idx in results_list2[0]['ablation_only'].index:
    templist1 = []
    templist2 = []
    templist3 = []
    templist4 = []
    templist5 = []
    for i in range(100):
        templist1.append(results_list2[i]['ablation_only'].loc[idx, 'error_distance'][0])
        templist2.append(results_list2[i]['ablation_only'].loc[idx, 'error_distance'][1])
        temp = results_list2[i]['ablation_only'].loc[idx, 'R2']
       
        templist3.append(temp)
        templist4.append(np.mean(temp, axis=0))
        templist5.append(np.std(temp, axis=0))
    blocking_df.loc[idx, 'error_mean'] = templist1
    blocking_df.loc[idx, 'error_std'] = templist2
    blocking_df.loc[idx, 'R2'] = templist3
    blocking_df.loc[idx, 'R2-dim3_mean'] = templist4
    blocking_df.loc[idx, 'R2-dim3_std'] = templist5
        
# print(blocking_df)

blocking_df_summary = pd.DataFrame(columns=['error_mean_mean', 'error_mean_std'])
for idx in blocking_df.index:
    blocking_df_summary.loc[idx, 'error_mean_mean'] = \
        '%.4f' % np.mean(np.array(blocking_df.loc[idx, 'error_mean']))
    blocking_df_summary.loc[idx, 'error_mean_std'] = \
        '%.4f' % np.std(np.array(blocking_df.loc[idx, 'error_mean']))
    blocking_df_summary.loc[idx, 'R2_mean'] = \
        '%.4f' % np.mean(np.array(blocking_df.loc[idx, 'R2-dim3_mean']), axis=0)
    blocking_df_summary.loc[idx, 'R2_std'] = \
        '%.4f' % np.std(np.array(blocking_df.loc[idx, 'R2-dim3_mean']), axis=0)
  
print(blocking_df_summary)


#%% adjusting connection weights
from analyses.blocking import blocking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SP, SPText, nSP = ts.SP, ts.SPText, ts.nSP 
color_map_sp, mypalsp = ts.color_map_sp, ts.mypalsp

# 
def run_adjusting(i):

    input_size = 5
    hidden_size = 200
    output_size = 2
    
    load_name = 'rnn_Ca1_%d' % i
    load_path = PATH_LOAD_models + os.sep + load_name +  '.pth'
    model = rs.rRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))     
    
    dict_mol = results_list[i]['modulation']['dict_mol']
    
    init_state = (torch.zeros((1, hidden_size))).to('cpu')
    test_cd = results_list[i]['test']
    inputs_here = results_list[i]['test'].inputs('Ca1')
    x_tensor = torch.from_numpy(np.array(inputs_here, dtype=np.float32))
    
    period = 'MO'
    adjusting_df = pd.DataFrame(columns=['error_distance', 'correct_ratio',
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
                        [i for i in dict_mol[period][to_mod]], ia] *= 1.5
                                                  
            h1, r1, pred1 = model_ia(x_tensor, init_state)
            
            p1_np = pred1.detach().numpy()
            r1_np = r1.detach().numpy()
            # target_pos = test_cd.object_dict['target_pos']
            # hand_pos1 = helper.getting_position_from_velocity(p1_np, test_cd)
          
            
            adjusting_dict, pcs1, evr1, SPIdx1, MDIdx1 = blocking(
                r1_np, p1_np, test_cd, SPText)
            adjusting_df.loc['%s to %s' % (from_mod, to_mod), :] = adjusting_dict

    return adjusting_df


adjusting_list = []
for i in range(100):  
    print(i)                  
    adjusting_list.append(run_adjusting(i))

adjusting_summary = pd.DataFrame(columns=['error_mean', 'error_std', 'R2', 
                                         'R2-dim3_mean', 'R2-dim3_std'])
for idx in adjusting_list[0].index:
    templist1 = []
    templist2 = []
    templist3 = []
    templist4 = []
    templist5 = []
    for i in range(100):
        templist1.append(adjusting_list[i].loc[idx, 'error_distance'][0])
        templist2.append(adjusting_list[i].loc[idx, 'error_distance'][1])
        temp = adjusting_list[i].loc[idx, 'R2']
        temp_np = temp
        templist3.append(np.mean(temp_np))

    adjusting_summary.loc[idx, 'error_mean'] = templist1
    adjusting_summary.loc[idx, 'error_std'] = templist2
    adjusting_summary.loc[idx, 'R2'] = templist3
        
print(adjusting_summary)
for idx in adjusting_summary.index:
    print('%s: error mean=%.4f, std=%.4f, R2 mean=%.4f, std=%.4f'
          % (idx, 
             np.mean(adjusting_summary.loc[idx, 'error_mean']),
             np.std(adjusting_summary.loc[idx, 'error_mean']),
             np.mean(adjusting_summary.loc[idx, 'R2']),
             np.std(adjusting_summary.loc[idx, 'R2'])))

adjusting_summary.to_csv(os.path.join(PATH_SAVE, 'adjusting_summary.csv'))

#%%
import scikit_posthocs as sp

stat, p = stats.kruskal(*[adjusting_summary.loc[i, 'R2'] 
                         for i in adjusting_summary.index])
print(p)
p_dunn = sp.posthoc_dunn([adjusting_summary.loc[i, 'R2'] 
                         for i in adjusting_summary.index], p_adjust='holm')
print(p_dunn)



stat, p = stats.kruskal(*[adjusting_summary.loc[i, 'error_mean'] 
                         for i in adjusting_summary.index])
print(p)
p_dunn = sp.posthoc_dunn([adjusting_summary.loc[i, 'error_mean'] 
                         for i in adjusting_summary.index], p_adjust='holm')
print(p_dunn)
