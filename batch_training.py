import os
import numpy as np
import torch
import settings.RNN_setting as rs
import settings.task_setting as ts
import settings.training_setting as trs
# from multiprocessing import Process, cpu_count, Pool


PATH_SAVE = r'/AMAX/cuihe_lab/chenyun/code/CR_version_spyder/models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'


# set parameters
input_size = 5
hidden_size = 200
output_size = 2

input_type = 'Ca1' #'C1', 'a1'

n_steps = 500
n_train = 100


def func(i):
    model = rs.rRNN(input_size, hidden_size, output_size).to(device)
    generate = ts.StandardCondition
    save_name = 'rnn_%s_%d' % (input_type, i)
    
    st = trs.training(model, n_steps, n_train, generate, input_type, 
                      PATH_SAVE+os.sep+save_name, device)
    np.save(PATH_SAVE+os.sep+save_name+'.npy', st)


if __name__ == '__main__':
    
    # ctx = torch.multiprocessing.get_context("spawn")
    # p = ctx.Pool(200)
    
    # p.map(func, range(10))
    # p.close()
    # p.join()

    for i in np.arange(100):
        print(i)
        func(i)
    
    
