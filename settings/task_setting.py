# -*- coding: utf-8 -*-
"""

To define basic setup for paradigm and to generate analog signals as input and output templates.
"""

import numpy as np
from matplotlib.colors import ListedColormap
import helpers.helper_functions as hf

# Task-related public variables
# unit: m
R_CIRCLE = 0.15
OFF_SCREEN_DISTANCE = 0.05

# unit: ms
time_delay = 400
time_reaction = 200
time_move = 150

TO, GO, MO = 0, time_delay, time_delay + time_reaction

time_total = time_delay + time_reaction + time_move
# unit: s
time_points = np.arange(0, time_total)/1000


input_label_dict = {'a': ['position-x', 'position-y'],
                    'b': ['cos(theta_t)'],
                    'A': ['shifting-intention-x', 'shifting-intention-y'],
                    'B': ['afterGO-intention-x', 'afterGO-intention-y'],
                    'C': ['50BeforeMO-intention-x', '50beforeMO-intention-y'],
                    '1': ['GO-step-off_on'],
                    '2': ['GO-step-on_off'],
                    '3': ['GO-pulse']}


SP = np.array([-240, -120, 0, 120, 240])/180*np.pi

# sp_dict = {'reaching': np.array([0]) / 180 * np.pi,
#            'inter-1sp': np.array([0, 120]) / 180 * np.pi,
#            'inter-5sp': np.array([-240, -120, 0, 120, 240]) / 180 * np.pi,
#            'inter-9sp': np.array(
#                [-360, -240, -180, -120, 0, 120, 180, 240, 360]) / 180 * np.pi}
# color_map_sp = plt.cm.get_cmap('rainbow', nSP)

SPText = [str(int(i/np.pi*180)) for i in np.unique(SP)]
nSP = len(SPText)
color_map_sp = tuple(map(tuple, np.array([[127, 41, 211, 255],
                                          [36, 158, 195, 255],
                                          [81, 219, 137, 255],
                                          [224, 166, 102, 255],
                                          [211, 40, 40, 255]])/255))
color_map_sp = ListedColormap(color_map_sp, name='color_sp')
mypalsp = {i: color_map_sp(i) for i in range(nSP)}


MD = np.arange(0, 360, 45)
MDText = MD.astype(str)
nMD = len(MDText)
color_map_md = tuple(map(tuple, np.array([[3, 3, 3, 255],
                                          [80, 82,85,255],
                                          [155, 160, 166, 255],
                                          [220, 217, 213, 255],
                                          [250, 201,139, 255],
                                          [207, 158, 101, 255],
                                          [148, 107, 68, 255],
                                          [92, 59, 37, 255]])/255))
color_map_md = ListedColormap(color_map_md, name='color_md')
mypalmd = {i: color_map_md(i) for i in range(nMD)}


def input_func(time_dict, object_dict, typename):

    # unzip variables
    time_total_random = time_dict['time_total']
    go_random, mo_random = time_dict['go'], time_dict['mo']
    rmt = time_total_random - go_random
    num_trial = len(time_total_random)

    (target_pos, target_rad, target_pre) = (object_dict['target_pos'], 
                                            object_dict['target_rad'], 
                                            object_dict['target_pre'])

    # package inputs into a dict
    input_dict = {}

    if 'a' in typename:
        # moving-target position in 2-D orthogonal coordinates
        input_dict['a'] = target_pre

    if 'b' in typename:
        target_rad_cos = np.cos(target_rad)
        input_dict['b'] = target_rad_cos

    if 'A' in typename:
        motor_intention_always = np.zeros_like(target_pos)
        for i_trial in range(num_trial):
            motor_intention_always[i_trial, :go_random[i_trial], :] = (
                target_pos[i_trial, 
                           rmt[i_trial]:go_random[i_trial] + rmt[i_trial], :])

            motor_intention_always[i_trial, 
                                   go_random[i_trial]:mo_random[i_trial], 
                                   :] = (
                target_pos[i_trial, time_total_random[i_trial] - 1, :])

            motor_intention_always[i_trial, mo_random[i_trial]:, :] = 0
        input_dict['A'] = motor_intention_always

    if 'B' in typename:
        motor_intention_after = np.zeros_like(target_pos)
        for i_trial in range(num_trial):
            motor_intention_after[i_trial, 
                                  go_random[i_trial]:mo_random[i_trial], 
                                  :] = (
                target_pos[i_trial, time_total_random[i_trial] - 1, :]
            )
        input_dict['B'] = motor_intention_after
    
    if 'C' in typename:
        motor_intention_after = np.zeros_like(target_pos)
        for i_trial in range(num_trial):
            motor_intention_after[i_trial, 
                                  mo_random[i_trial]-50:mo_random[i_trial], 
                                  :] = (
                target_pos[i_trial, time_total_random[i_trial] - 1, :]
            )
        input_dict['C'] = motor_intention_after
    

    if '1' in typename:
        motor_trigger_step_on = np.zeros_like(target_rad)
        for i_trial in range(num_trial):
            motor_trigger_step_on[i_trial, go_random[i_trial]:, :] = 1
        input_dict['1'] = motor_trigger_step_on

    if '2' in typename:
        motor_trigger_step_off = np.zeros_like(target_rad)
        for i_trial in range(num_trial):
            motor_trigger_step_off[i_trial, :go_random[i_trial], :] = 1
        input_dict['2'] = motor_trigger_step_off

    if '3' in typename:
        motor_trigger_pulse = np.zeros_like(target_rad)
        for i_trial in range(num_trial):
            sigma_for_go = hf.calculate_best_sigma_for_gaussian(
                np.arange(time_total_random[i_trial]),
                go_random[i_trial],
                go_random[i_trial]+10)

            motor_trigger_pulse[i_trial, 
                                go_random[i_trial]:go_random[i_trial]+10, 
                                :] = \
                (hf.gaussian_p(np.arange(time_total_random[i_trial]), 
                               go_random[i_trial], go_random[i_trial]+10,
                               sigma_for_go))[:, np.newaxis]
            motor_trigger_pulse[i_trial, 
                                go_random[i_trial]-10:go_random[i_trial], 
                                :] = \
                (hf.gaussian_p(np.arange(time_total_random[i_trial]), 
                               go_random[i_trial], go_random[i_trial]-10,
                               sigma_for_go))[:, np.newaxis]
        input_dict['3'] = motor_trigger_pulse

    # transform dict to array
    input_pkg = np.dstack([input_dict[i] for i in list(typename)])

    return input_pkg


def output_func(time_dict, object_dict):

    time_total_random = time_dict['time_total']
    time_move_random = time_dict['time_move']
    mo = time_dict['mo']
    num_trial = len(time_total_random)

    target_rad = object_dict['target_rad']

    motor_target = np.zeros((num_trial, max(time_total_random), 2))

    for i_trial in range(num_trial):
        v_best, desired_v, desired_p = hf.cal_desired_traj(
            np.arange(time_move_random[i_trial]), R_CIRCLE, _tau_reach=40)
        motor_target[i_trial, mo[i_trial]:time_total_random[i_trial], :] = \
            np.vstack([
                np.cos(target_rad[i_trial, time_total_random[i_trial] - 1, 0]) 
                * desired_v * 10 ** 3,
                np.sin(target_rad[i_trial, time_total_random[i_trial] - 1, 0]) 
                * desired_v * 10 ** 3]).T

    return motor_target


class DelayedGenerator:
    def __init__(self, target_init, target_speed):

        self.num_trial = len(target_init)
        self.condition_dict = {'target_init': target_init,
                               'target_speed': target_speed}
        self.time_dict = {}
        self.object_dict = {}

    def time_assembling(self, func, **kwargs):

        time_delay_random, time_reaction_random, time_move_random = \
            func(**kwargs)
        num_trial = self.num_trial

        # sort the total time and then sort other periods correspondingly
        # necessary for training
        time_total_random = (time_delay_random 
                             + time_reaction_random 
                             + time_move_random)
        decreasing_idx = np.flip(np.argsort(time_total_random))
        time_total_random = time_total_random[decreasing_idx]

        time_move_random = time_move_random[decreasing_idx]
        time_reaction_random = time_reaction_random[decreasing_idx]
        time_delay_random = time_delay_random[decreasing_idx]

        # mark important time points
        go_random = time_delay_random
        mo_random = time_delay_random + time_reaction_random
        rmt = time_reaction_random + time_move_random

        assert (go_random + rmt == time_total_random).all()
        assert (mo_random + time_move_random == time_total_random).all()

        # calculate target's radium and position for every time point
        time_points_random = [np.arange(time_total_random[i]) 
                              for i in range(num_trial)]

        self.time_dict = {'time_total': time_total_random,
                          'go': go_random, 'mo': mo_random,
                          'time_points': time_points_random,
                          'time_move': time_move_random}

    def object_setting(self):

        num_trial = self.num_trial
        target_init = self.condition_dict['target_init']
        target_speed = self.condition_dict['target_speed']

        time_total_random = self.time_dict['time_total']
        time_points_random = self.time_dict['time_points']
        max_time_total = max(time_total_random)

        target_init = np.mod(target_init, np.pi*2)
        target_rad = np.zeros((num_trial, max_time_total, 1))
        for i_trial in range(num_trial):
            target_rad[i_trial, 0:time_total_random[i_trial], 0] = (
                target_speed[i_trial, 0]*time_points_random[i_trial]/1000
                + target_init[i_trial, 0])
        target_rad = np.mod(target_rad, np.pi*2)

        target_pos = R_CIRCLE * np.dstack([np.cos(target_rad), 
                                           np.sin(target_rad)])
        for i_trial in range(num_trial):
            target_pos[i_trial, time_total_random[i_trial]:, :] = 0

        # package useful variables
        self.object_dict = {'target_rad': target_rad,
                            'target_pos': target_pos,
                            'target_pre': target_pos}

    def inputs(self, tp):
        return input_func(self.time_dict, self.object_dict, typename=tp)

    def outputs(self):
        return output_func(self.time_dict, self.object_dict)


class StandardCondition(DelayedGenerator):
    def __init__(self, target_init, target_speed):
        super().__init__(target_init, target_speed)

    def time_setting(self):

        num_trial = self.num_trial

        # set random delay time, reaction time, and movement time
        time_delay_random = np.random.randint(200, size=num_trial) + time_delay
        time_reaction_random = ((3 * np.random.randn(num_trial)).astype(int) 
                                + time_reaction)
        time_move_random = ((5 * np.random.randn(num_trial)).astype(int) 
                            + time_move)

        return time_delay_random, time_reaction_random, time_move_random

    def generating(self):
        super().time_assembling(self.time_setting)
        super().object_setting()

    def inputs(self, tp):
        self.generating()
        return input_func(self.time_dict, self.object_dict, typename=tp)


# update

class TimeSpecifiedCondition(DelayedGenerator):
    def __init__(self, target_init, target_speed, specified_dict):
        super().__init__(target_init, target_speed)

        self.specified_dict = specified_dict

    def time_setting(self):

        num_trial = self.num_trial
        time_delay_random = self.specified_dict['time_delay']
        time_reaction_random = self.specified_dict['time_reaction']
        time_move_random = self.specified_dict['time_move']

        assert len(time_delay_random) == num_trial
        assert len(time_reaction_random) == num_trial
        assert len(time_move_random) == num_trial

        return time_delay_random, time_reaction_random, time_move_random

    def generating(self):
        super().time_assembling(self.time_setting)

        time_delay_random = self.specified_dict['time_delay']
        time_reaction_random = self.specified_dict['time_reaction']
        time_move_random = self.specified_dict['time_move']
        time_total_random = time_delay_random + time_reaction_random + time_move_random
        decreasing_idx = np.flip(np.argsort(time_total_random))

        # Note: sorting because time and target must match in this case
        self.condition_dict['target_init'] = self.condition_dict['target_init'][decreasing_idx]
        self.condition_dict['target_speed'] = self.condition_dict['target_speed'][decreasing_idx]

        super().object_setting()

    def inputs(self, tp):
        self.generating()
        return input_func(self.time_dict, self.object_dict, typename=tp)