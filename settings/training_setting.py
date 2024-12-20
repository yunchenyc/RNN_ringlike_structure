# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import torch
from torch import nn
import tqdm
import settings.task_setting as ts


def training(model, n_steps, n_train, generate, input_type, save_name, device):

    # training setup
    learning_rate = 0.001
    alpha = 1e-7
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    mse_func = nn.MSELoss()
    loss_list = []

    with tqdm.tqdm(total=n_steps) as pbar:
        for step in range(n_steps):

            optimizer.zero_grad()

            sp_array = ts.SP[np.random.choice(range(len(ts.SP)), (n_train, 1))]
            ti_array = np.random.rand(n_train, 1)*np.pi*2

            conditions = generate(ti_array, sp_array)
            inputs_selected = conditions.inputs(input_type)
            output_template = conditions.outputs()

            x_tensor = torch.from_numpy(np.array(inputs_selected, dtype=np.float32))
            y_tensor = torch.from_numpy(np.array(output_template, dtype=np.float32))

            x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)

            # init_state = (torch.rand((1, model.hidden_size1))*0.3).to(device)
            init_state = (torch.zeros((1, model.hidden_size))).to(device)

            h, r, hand_v = model(x_tensor, init_state)

            # minimize the response
            regular = torch.square(r).sum()

            error = mse_func(hand_v, y_tensor) + alpha*regular

            error.requires_grad_(True)
            error.backward()
            optimizer.step()

            loss_list.append(error.cpu().detach().numpy())
            pbar.update(1)

            if error < 1e-3:
                break

    torch.save(model.state_dict(), save_name+'.pth')

    return {'nSteps': n_steps, 'nTrain': n_train,
            'input_type': input_type, 'save_name': save_name,
            'learning_rate': learning_rate, 'loss_list': loss_list,
            'training_set': (conditions,
                             x_tensor.cpu().detach().numpy(),
                             y_tensor.cpu().detach().numpy())}
