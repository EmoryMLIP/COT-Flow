import os
import numpy as np
import pandas as pd

loss = pd.read_csv('.../OT-Flow/experiments/cnf/tabcond/lv_valid_hist.csv').to_numpy()
loss_min = np.sort(loss[:, -1])[0]
indx = np.where(loss[:, -1] == loss_min)[-1]
param = pd.read_csv('.../OT-Flow/experiments/cnf/tabcond/lv_params_hist.csv').to_numpy()
param_train = param[indx].squeeze()

width = int(param_train[4])
batch_size = int(param_train[-1])
lr = param_train[5]
nt = int(param_train[3])
alpha = [1.0, param_train[1], param_train[2]]

os.system(
    "python trainOTflowCond.py --data 'lv' --dx 4 --num_epochs 1000 --drop_freq 0 --save_test False\
    --val_freq 50 --weight_decay 0.0 --prec single --early_stopping 20 --lr_drop 2.0 --batch_size " + \
    str(batch_size) + " --test_batch_size " + str(batch_size) + " --lr " + str(lr) + " --nt " + str(nt) + \
    " --nt_val " + str(32) + " --m " + str(width) + " --alph " + str(alpha[0]) + ',' + str(alpha[1]) + ',' + \
    str(alpha[2]) + " --save 'experiments/cnf/tabcond/lv'"
)

