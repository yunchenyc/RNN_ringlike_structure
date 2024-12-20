# -*- coding: utf-8 -*-
"""

helper functions for newIC
"""

import numpy as np
from scipy import optimize
from scipy import integrate


def cal_desired_traj(_t, _d_reach, _tau_reach=120):
    """
    Calculate desired trajectory (time-varying position).

    Parameters
    ----------
    _t: array, time bins
    _d_reach: float, reach distance
    _tau_reach: constant, optional
        time constant of reach velocity profile. The default is 120.

    Returns
    -------
    v_best: float
    desired_velocity: array of [nTime].
    desired_position: array of [nTime].

    """

    # define a common bell-shaped scalar speed profile
    def cal_v(t, v):
        return v * (t / _tau_reach) ** 2 * np.exp(-(t / _tau_reach) ** 2 / 2)

    # define the absolute distance error dependent on velocity
    def d2d_reach(v):
        return abs(integrate.quad(cal_v, min(_t), max(_t), args=(v,))[0] - _d_reach)

    # velocity is optimized so that the hand reaches the target
    res = optimize.minimize(d2d_reach, (np.array(0.0001)), method='SLSQP',
                            bounds=((0, 1e+2),))
    v_best = res.x

    # get _desired_velocity
    desired_velocity = cal_v(_t, v_best)

    # get _desired_position
    desired_position = []
    for it in _t:
        desired_position.append(
            integrate.quad(cal_v, _t[0], it, args=(v_best,))[0])

    desired_position = np.array(desired_position)

    return v_best, desired_velocity, desired_position


def gaussian_f(mu, sigma, x):
    # gaussian function
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


# calculate points in a Gaussian distribution from idt to idt2, idt at peak=1
def gaussian_p(t, idt, idt2, sigma):
    return gaussian_f(
        t[idt], sigma, t[min([idt, idt2]):max([idt, idt2])]) / max(
        gaussian_f(t[idt], sigma, t[min([idt, idt2]):max([idt, idt2])]))


def calculate_best_sigma_for_gaussian(ts, tp1, tp2):
    """
    Calculate the sigma of a Gaussian distribution covering the given period.

    Parameters
    ----------
    ts: array, time points.
    tp1: int, time point 1 (start point)
    tp2: int, time point 2 (end point)

    Returns
    -------
    best_sigma: float, sigma optimized for the given period
    desired_velocity_: array of [nTime, 2], (x, y).
    desired_position_: array of [nTime, 2], (x, y).

    """

    def cost_function(v):
        return abs(min(gaussian_p(ts, tp1, tp2, v)) - 1e-3)

    res = optimize.minimize(cost_function, (np.array(20)), method='Powell')
    best_sigma = res.x

    return best_sigma


def get_direction_split_index(direction, split_num):
    """
    Note!!!
    Here direction is split into 'split_num' parts, which center at the
    2*pi/split_num, e.g. the first part centers at 0 while the second part
    center at pi/4 when the 'split_num=8'
    """
    direction = np.mod(direction, np.pi * 2)
    direction_space = np.pi * 2 / split_num
    direction_idx = direction // direction_space
    direction_idx[direction_idx == split_num + 1] = 0

    return direction_idx.astype(int)


# @title @getting_postion_from_velocity

def getting_position_from_velocity(handv_np, cd):
    pos_np = np.zeros_like(handv_np)
    for iTrial in range(handv_np.shape[0]):
        for iOutput in range(2):
            for iTime in np.arange(cd.time_dict['time_move'][iTrial]):
                pos_np[iTrial, cd.time_dict['mo'][iTrial] + iTime, iOutput] = \
                    integrate.trapz(
                        handv_np[iTrial, 
                                 cd.time_dict['mo'][iTrial]
                                 :cd.time_dict['mo'][iTrial] + iTime + 1, 
                                 iOutput],
                    np.arange(iTime + 1) / 1000)

    return pos_np
