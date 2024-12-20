#%%
import os
import numpy as np
import pandas as pd
import pickle

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


#%% Modulation
mo_modul_list = [results_list[i]['modulation']['mo_modul_np']
                 for i in range(100)]
mo_modul_df = pd.DataFrame(columns=['G', 'S', 'A', 'G+S', 'S+A', 'G+A',
                                    'G+S+A', 'no modul / ac', 'no ac / all'])
# S, G, A in mo_modul_np
for i in range(100):
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

