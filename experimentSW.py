import os
import numpy as np
import pandas as pd

# grab csvs
loss = pd.read_csv('.../OT-Flow/experiments/cnf/tabcond/sw_valid_hist.csv').to_numpy()
loss_min = np.sort(loss[:, -1])[0]
indx = np.where(loss[:, -1] == loss_min)[-1]
param = pd.read_csv('.../OT-Flow/experiments/cnf/tabcond/sw_params_hist.csv').to_numpy()
param_train = param[indx].squeeze()

# load hyperparameters
alpha = [1.0, param_train[1], param_train[2]]
nt = int(param_train[3])
m = int(param_train[4])
my = int(param_train[5])
myout = int(param_train[6])
lr = param_train[7]
batch_size = int(param_train[8])
num_steps = int(param_train[-1])

os.system(
    "python trainOTflowSW.py --dx 14 --num_epochs 1000 --num_steps " + str(num_steps) + " --drop_freq 0\
     --val_freq 70 --weight_decay 0.0 --prec single --early_stopping 20 --lr_drop 2.0 --batch_size " +
    str(batch_size) + " --test_batch_size " + str(batch_size) + " --lr " + str(lr) + " --nt " + str(nt) +
    " --nt_val " + str(32) + " --m " + str(m) + " --m_y " + str(my) + " --mout_y " + str(myout) + " --alph " + \
    str(alpha[0]) + ',' + str(alpha[1]) + ',' + str(alpha[2]) + " --save 'experiments/cnf/tabcond/sw'"
)
