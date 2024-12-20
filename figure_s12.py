#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import settings.RNN_setting as rs
import torch
import seaborn as sns
import scikit_posthocs as sp

PATH_LOAD_models = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/models')
PATH_LOAD_results = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/results')
PATH_SAVE = format(r'/AMAX/cuihe_lab/chenyun/code/RNN_ringlike_geometry/outputs')

#%% load individual results 
results_list = []
for i in range(100):

    load_name = 'rnn_Ca1_%d' % i

    with open(os.path.join(PATH_LOAD_results, '%s_results2.pkl' % load_name), 'rb') as f:
        results = pickle.load(f)
        results_list.append(results)


#%% decoding
### Figure S12A ###

decoding_df = pd.DataFrame(columns=['error_mean', 'error_std', 'R2', 
                                    'R2-dim3_mean', 'R2-dim3_std'])

sp_decoding = []
md_decoding = []
for i in range(100): 
    sp_decoding.append(results_list[i]['decoding']['sp'])
    md_decoding.append(results_list[i]['decoding']['md'])

sp_to = np.array([i[0] for i in sp_decoding]).mean(axis=1)
sp_go = np.array([i[1] for i in sp_decoding]).mean(axis=1)
sp_mo = np.array([i[2] for i in sp_decoding]).mean(axis=1)

md_to = np.array([i[0] for i in md_decoding]).mean(axis=1)
md_go = np.array([i[1] for i in md_decoding]).mean(axis=1)
md_mo = np.array([i[2] for i in md_decoding]).mean(axis=1)

plt.figure(figsize=(6, 1.5))
plt.plot(np.arange(50, 300, 50), np.mean(md_to, axis=0), 'k-')
plt.fill_between(np.arange(50, 300, 50), 
                 np.mean(md_to, axis=0)-np.std(md_to, axis=0),
                 np.mean(md_to, axis=0)+np.std(md_to, axis=0),
                 facecolor='k', alpha=0.2)
plt.plot(np.arange(50, 300, 50), np.mean(sp_to, axis=0), '-', color=np.array([0, 115, 189])/255)
plt.fill_between(np.arange(50, 300, 50), 
                 np.mean(sp_to, axis=0)-np.std(sp_to, axis=0),
                 np.mean(sp_to, axis=0)+np.std(sp_to, axis=0),
                 facecolor=np.array([0, 115, 189])/255, alpha=0.2)
plt.hlines(y=0.2, xmin=50, xmax=250, colors=np.array([0, 115, 189])/255, linestyles='--')
plt.hlines(y=0.125, xmin=50, xmax=250, colors='k', ls='--')

plt.plot(np.arange(350, 900, 50), np.mean(md_go, axis=0), 'k-')
plt.fill_between(np.arange(350, 900, 50), 
                 np.mean(md_go, axis=0)-np.std(md_go, axis=0),
                 np.mean(md_go, axis=0)+np.std(md_go, axis=0),
                 facecolor='k', alpha=0.2)
plt.plot(np.arange(350, 900, 50), np.mean(sp_go, axis=0), '-', color=np.array([0, 115, 189])/255)
plt.fill_between(np.arange(350, 900, 50), 
                 np.mean(sp_go, axis=0)-np.std(sp_go, axis=0),
                 np.mean(sp_go, axis=0)+np.std(sp_go, axis=0),
                 facecolor=np.array([0, 115, 189])/255, alpha=0.2)
plt.hlines(y=0.2, xmin=350, xmax=850, colors=np.array([0, 115, 189])/255, linestyles='--')
plt.hlines(y=0.125, xmin=350, xmax=850, colors='k', ls='--')

plt.plot(np.arange(950, 1300, 50), np.mean(md_mo, axis=0), 'k-')
plt.fill_between(np.arange(950, 1300, 50), 
                 np.mean(md_mo, axis=0)-np.std(md_mo, axis=0),
                 np.mean(md_mo, axis=0)+np.std(md_mo, axis=0),
                 facecolor='k', alpha=0.2)
plt.plot(np.arange(950, 1300, 50), np.mean(sp_mo, axis=0), '-', color=np.array([0, 115, 189])/255)
plt.fill_between(np.arange(950, 1300, 50), 
                 np.mean(sp_mo, axis=0)-np.std(sp_mo, axis=0),
                 np.mean(sp_mo, axis=0)+np.std(sp_mo, axis=0),
                 facecolor=np.array([0, 115, 189])/255, alpha=0.2)
plt.hlines(y=0.2, xmin=950, xmax=1250, colors=np.array([0, 115, 189])/255, linestyles='--')
plt.hlines(y=0.125, xmin=950, xmax=1250, colors='k', ls='--')

plt.xticks([0, 200, 400, 600, 800, 1000, 1200],
            labels=['TO', 200, -200, 'GO', 200, -200, 'MO'])
plt.ylim(-0.02, 1.1)

plt.savefig(os.path.join(PATH_SAVE, 'svm_decoding_step50ms.png'), dpi=300)
plt.show() 


#%% input and output connection weight
input_size = 5
hidden_size = 200
output_size = 2

period = 'MO'

weight_df = pd.DataFrame(columns=['weight', 'absweight', 'from', 'to'])

for i in range(100):
    load_name = 'rnn_Ca1_%d' % i
    load_path = PATH_LOAD_models + os.sep + load_name +  '.pth'
    model = rs.rRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))     

    dict_mol = results_list[i]['modulation']['dict_mol']

    for mod in ['S', 'G', 'A']:
        
        nomod = [i for i in ['S', 'G', 'A'] if i !=mod]
       
        for iia, ia in enumerate([[0, 1], [2, 3], 4]):
            temp = model.state_dict()['cell.w_ih.weight'][
                    [i for i in dict_mol[period][mod] 
                     if (i not in dict_mol[period][nomod[0]])&
                     (i not in dict_mol[period][nomod[1]])], :].numpy()
            maxtemp = abs(model.state_dict()['cell.w_ih.weight'][:, ia].numpy()).max()
            temp = temp/maxtemp
            if isinstance(ia, list):
                temp = np.mean(temp[:, ia], axis=1)
            else:
                temp = temp[:, ia]
                
            for ii in temp:
                weight_df = weight_df._append({'weight': ii, 
                                               'absweight': abs(ii), 
                                               'from': iia, 'to': mod},
                                              ignore_index=True)
        
        temp = model.state_dict()['cell.w_ho.weight'][
                        :, [i for i in dict_mol[period][mod] 
                            if (i not in dict_mol[period][nomod[0]])&
                            (i not in dict_mol[period][nomod[1]])]].numpy()
        maxtemp = abs(model.state_dict()['cell.w_ho.weight'].numpy()).max()
        temp = temp/maxtemp
        temp = np.mean(temp, axis=0)
        for jj in temp:
            weight_df = weight_df._append({'weight': jj, 
                                           'absweight': abs(jj), 
                                           'from': mod, 'to': 'output'},
                                          ignore_index=True)

#%%
### Figure S12B ###
idx111 = [i for i in weight_df.index 
          if (weight_df.loc[i, 'to'] in ['S', 'G', 'A']) &
          (weight_df.loc[i, 'weight']>0)]
plot_df = weight_df.loc[idx111]
plt.figure(dpi=300)
sns.boxplot(data=plot_df, x='from', y='weight', hue='to')
plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.savefig(os.path.join(PATH_SAVE, 'in_weight_positive.png'), dpi=300)
plt.show()

for inputi, inputtype in enumerate(['motor intention', 'target location', 'Go signal']):
    print(inputtype)
    stat, p = stats.kruskal(
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='S'), 'weight'].values,  
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='G'), 'weight'].values,
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='A'), 'weight'].values)
    print('K-W test p=%f' % p)
    p_dunn = sp.posthoc_dunn([
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='S'), 'weight'].values,  
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='G'), 'weight'].values,
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='A'), 'weight'].values], 
        p_adjust='holm')
    print(p_dunn)


idx111 = [i for i in weight_df.index 
          if (weight_df.loc[i, 'to'] in ['S', 'G', 'A']) &
          (weight_df.loc[i, 'weight']<0)]
plot_df = weight_df.loc[idx111]
plt.figure(dpi=300)
sns.boxplot(data=plot_df, x='from', y='weight', hue='to')
plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.savefig(os.path.join(PATH_SAVE, 'in_weight_negative.png'), dpi=300)
plt.show()

for inputi, inputtype in enumerate(['motor intention', 'target location', 'Go signal']):
    print(inputtype)
    stat, p = stats.kruskal(
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='S'), 'weight'].values,  
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='G'), 'weight'].values,
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='A'), 'weight'].values)
    print('K-W test p=%f' % p)
    p_dunn = sp.posthoc_dunn([
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='S'), 'weight'].values,  
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='G'), 'weight'].values,
        plot_df.loc[(plot_df['from']==inputi)&(plot_df['to']=='A'), 'weight'].values], 
        p_adjust='holm')
    print(p_dunn)


#%% 
### Figure S12C ###
idx222 = [i for i in weight_df.index 
          if (weight_df.loc[i, 'from'] in ['S', 'G', 'A']) &
          (weight_df.loc[i, 'weight']>0)]
plot_df2 = weight_df.loc[idx222]
plt.figure(dpi=300)
sns.boxplot(data=plot_df2, x='from', y='weight', hue='to')
plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.ylim(-0.05, 1.05)
plt.savefig(os.path.join(PATH_SAVE, 'out_weight_positive.png'), dpi=300)
plt.show()
stat, p = stats.kruskal(
    plot_df2.loc[(plot_df2['from']=='S')&(plot_df2['to']=='output'), 'weight'].values,  
    plot_df2.loc[(plot_df2['from']=='G')&(plot_df2['to']=='output'), 'weight'].values,
    plot_df2.loc[(plot_df2['from']=='A')&(plot_df2['to']=='output'), 'weight'].values)
print('K-W test p=%f' % p)
p_dunn = sp.posthoc_dunn([
    plot_df2.loc[(plot_df2['from']=='S')&(plot_df2['to']=='output'), 'weight'].values,  
    plot_df2.loc[(plot_df2['from']=='G')&(plot_df2['to']=='output'), 'weight'].values,
    plot_df2.loc[(plot_df2['from']=='A')&(plot_df2['to']=='output'), 'weight'].values], 
    p_adjust='holm')
print(p_dunn)


idx222 = [i for i in weight_df.index 
          if (weight_df.loc[i, 'from'] in ['S', 'G', 'A']) &
          (weight_df.loc[i, 'weight']<0)]
plot_df2 = weight_df.loc[idx222]
plt.figure(dpi=300)
sns.boxplot(data=plot_df2, x='from', y='weight', hue='to')
plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.ylim(-1.05, 0.05)
plt.savefig(os.path.join(PATH_SAVE, 'out_weight_negative.png'), dpi=300)
plt.show()
stat, p = stats.kruskal(
    plot_df2.loc[(plot_df2['from']=='S')&(plot_df2['to']=='output'), 'weight'].values,  
    plot_df2.loc[(plot_df2['from']=='G')&(plot_df2['to']=='output'), 'weight'].values,
    plot_df2.loc[(plot_df2['from']=='A')&(plot_df2['to']=='output'), 'weight'].values)
print('K-W test p=%f' % p)

p_dunn = sp.posthoc_dunn([
    plot_df2.loc[(plot_df2['from']=='S')&(plot_df2['to']=='output'), 'weight'].values,  
    plot_df2.loc[(plot_df2['from']=='G')&(plot_df2['to']=='output'), 'weight'].values,
    plot_df2.loc[(plot_df2['from']=='A')&(plot_df2['to']=='output'), 'weight'].values], 
    p_adjust='holm')
print(p_dunn)
