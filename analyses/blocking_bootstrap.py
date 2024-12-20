#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:18:34 2024

@author: chenyun
"""
import numpy as np
import pandas as pd
import random
import helpers.helper_functions as helper
import analyses.states as st
import analyses.geometry as ge


def blocking(r1_np, p1_np, test_cd, SPText):

    blocking_dict = {}
    target_pos = test_cd.object_dict['target_pos']
    hand_pos1 = helper.getting_position_from_velocity(p1_np, test_cd)
    
    err_dist = []
    for itrial in range(hand_pos1.shape[0]):
        temp = np.linalg.norm(
            hand_pos1[itrial, 
                     test_cd.time_dict['time_total'][itrial] - 1, :]
            - target_pos[itrial, 
                         test_cd.time_dict['time_total'][itrial] - 1, :])
        err_dist.append(temp)
    err_dist = np.array(err_dist)
    error_mean, error_std =  np.mean(err_dist), np.std(err_dist)
    correct_ratio = np.sum(err_dist < 0.02) / len(err_dist)
    
    blocking_dict['error_distance'] = [error_mean, error_std]
    blocking_dict['correct_ratio'] = correct_ratio
    
    if True: #correct_ratio > 0.5:
        # corr_idx = range(len(err_dist))
        corr_idx = np.argwhere(err_dist<0.02).flatten().tolist()
        corr_r1_np = r1_np[corr_idx, :, :]
        touch_direction = []
        for iTrial in corr_idx:
            endidx = test_cd.time_dict['time_total'][iTrial] - 1
            touch_direction.append(np.arctan2(hand_pos1[iTrial, endidx, 1],
                                              hand_pos1[iTrial, endidx, 0]))
        
        # state geometry
        pcs, evr = st.getting_neural_states_withPCA(
            corr_r1_np, -100, 0, 'mo', 2, test_cd)
        
        geometry_df = pd.DataFrame(columns=['pcs_points', 
                                            'ellipses', 'e-R2'])
        SPList, SPIdx, SPNum = np.unique(
            test_cd.condition_dict['target_speed'][corr_idx, 0],
            return_inverse=True, return_counts=True)
        assert len(SPList) == len(SPText)
            
        for isp in np.unique(SPIdx):
            pcs_points = (pcs[SPIdx==isp, 0], 
                          pcs[SPIdx==isp, 1],
                          pcs[SPIdx==isp, 2])
            
            el_points, pel_points, elparams = \
                ge.getting_fitted_ellipses(pcs_points, 'ellipses')
        
            r2 = ge.r_square(np.array(pel_points), np.array(pcs_points))
            
            geometry_df.loc[SPText[isp], 'pcs_points'] = pcs_points
            geometry_df.loc[SPText[isp], 'ellipses'] = (el_points, 
                                                        pel_points, 
                                                        elparams)
            geometry_df.loc[SPText[isp], 'e-R2'] = r2
            
            
        # calculate the MD clusters
        MDIdx = helper.get_direction_split_index(touch_direction, 8)
        mMDIdx = helper.get_direction_split_index(touch_direction, 36)
        mMDList, _ = np.unique(mMDIdx, return_counts=True)
        
        withinmean = st.calculating_distance_by_group(pcs, mMDIdx)
        # print('withinmean = %.3f' % withinmean)
        
        randommdidx = random.sample(list(mMDIdx), len(mMDIdx))
        controlmean = st.calculating_distance_by_group(pcs, randommdidx)
        # print('conmean = %.3f' % controlmean)
    
       
        blocking_dict['R2'] = geometry_df['e-R2'].values
            
        blocking_dict['within_mean'] = withinmean
       
        blocking_dict['control_mean'] = controlmean
    
    
    return blocking_dict, pcs, evr, SPIdx, MDIdx
    