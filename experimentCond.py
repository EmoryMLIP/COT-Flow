import os
import numpy as np
import pandas as pd

# TODO: Change to correct paths

loss_con = pd.read_csv('.../COT-Flow/experiments/cnf/tabcond/concrete_valid_hist.csv').to_numpy()
loss_enr = pd.read_csv('.../COT-Flow/experiments/cnf/tabcond/energy_valid_hist.csv').to_numpy()
loss_yat = pd.read_csv('.../COT-Flow/experiments/cnf/tabcond/yacht_valid_hist.csv').to_numpy()
loss_con_min = np.sort(loss_con[:, -1])[:10]
loss_enr_min = np.sort(loss_enr[:, -1])[:10]
loss_yat_min = np.sort(loss_yat[:, -1])[:10]
loss_con_min = loss_con_min.reshape(10, 1)
loss_enr_min = loss_enr_min.reshape(10, 1)
loss_yat_min = loss_yat_min.reshape(10, 1)
indx_con = np.where(loss_con[:, -1] == loss_con_min)[-1]
indx_enr = np.where(loss_enr[:, -1] == loss_enr_min)[-1]
indx_yat = np.where(loss_yat[:, -1] == loss_yat_min)[-1]

param_con = pd.read_csv('.../COT-Flow/experiments/cnf/tabcond/concrete_params_hist.csv').to_numpy()
param_enr = pd.read_csv('.../COT-Flow/experiments/cnf/tabcond/energy_params_hist.csv').to_numpy()
param_yat = pd.read_csv('.../COT-Flow/experiments/cnf/tabcond/yacht_params_hist.csv').to_numpy()
param_con_list = param_con[indx_con]
param_enr_list = param_enr[indx_enr]
param_yat_list = param_yat[indx_yat]


for i in range(10):
    for j in range(5):
        width_con = int(param_con_list[i, 4])
        width_enr = int(param_enr_list[i, 4])
        width_yat = int(param_yat_list[i, 4])

        batch_size_con = int(param_con_list[i, -1])
        batch_size_enr = int(param_enr_list[i, -1])
        batch_size_yat = int(param_yat_list[i, -1])

        lr_con = param_con_list[i, 5]
        lr_enr = param_enr_list[i, 5]
        lr_yat = param_yat_list[i, 5]

        nt_con = int(param_con_list[i, 3])
        nt_enr = int(param_enr_list[i, 3])
        nt_yat = int(param_yat_list[i, 3])

        alpha_con = [1.0, param_con_list[i, 1], param_con_list[i, 2]]
        alpha_enr = [1.0, param_enr_list[i, 1], param_enr_list[i, 2]]
        alpha_yat = [1.0, param_yat_list[i, 1], param_yat_list[i, 2]]

        os.system(
            "python trainOTflowCond.py --num_epochs 1000 --data 'concrete' --dx 1 --drop_freq 0 \
            --val_freq 20 --weight_decay 0.0 --prec single --early_stopping 10 --lr_drop 2.0 --batch_size " + \
            str(batch_size_con) + " --test_batch_size " + str(batch_size_con) + " --lr " + str(lr_con) + " --nt " + str(nt_con) + \
            " --nt_val " + str(32) + " --m " + str(width_con) + " --alph " + str(alpha_con[0]) + ',' + str(alpha_con[1]) + ',' + \
            str(alpha_con[2]) + " --save 'experiments/cnf/tabcond/concrete'"
        )

        os.system(
            "python trainOTflowCond.py --num_epochs 1000 --data 'energy' --dx 1 --drop_freq 0 \
            --val_freq 20 --weight_decay 0.0 --prec single --early_stopping 10 --lr_drop 2.0 --batch_size " + \
            str(batch_size_enr) + " --test_batch_size " + str(batch_size_enr) + " --lr " + str(lr_enr) + " --nt " + str(nt_enr) + \
            " --nt_val " + str(32) + " --m " + str(width_enr) + " --alph " + str(alpha_enr[0]) + ',' + str(alpha_enr[1]) + ',' + \
            str(alpha_enr[2]) + " --save 'experiments/cnf/tabcond/energy'"
        )

        os.system(
            "python trainOTflowCond.py --num_epochs 1000 --data 'yacht' --dx 1 --drop_freq 0 \
            --val_freq 20 --weight_decay 0.0 --prec single --early_stopping 10 --lr_drop 2.0 --batch_size " + \
            str(batch_size_yat) + " --test_batch_size " + str(batch_size_yat) + " --lr " + str(lr_yat) + " --nt " + str(nt_yat) + \
            " --nt_val " + str(32) + " --m " + str(width_yat) + " --alph " + str(alpha_yat[0]) + ',' + str(alpha_yat[1]) + ',' + \
            str(alpha_yat[2]) + " --save 'experiments/cnf/tabcond/yacht'"
        )

