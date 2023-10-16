import os
import numpy as np
import pandas as pd

# TODO: Change to correct paths

loss_pk = pd.read_csv('.../COT-Flow/experiments/cnf/tabjoint/parkinson_valid_hist.csv').to_numpy()
loss_rd = pd.read_csv('.../COT-Flow/experiments/cnf/tabjoint/rd_wine_valid_hist.csv').to_numpy()
loss_wt = pd.read_csv('.../COT-Flow/experiments/cnf/tabjoint/wt_wine_valid_hist.csv').to_numpy()
loss_pk_min = np.sort(loss_pk[:, -1])[:10]
loss_rd_min = np.sort(loss_rd[:, -1])[:10]
loss_wt_min = np.sort(loss_wt[:, -1])[:10]
loss_pk_min = loss_pk_min.reshape(10, 1)
loss_rd_min = loss_rd_min.reshape(10, 1)
loss_wt_min = loss_wt_min.reshape(10, 1)
indx_pk = np.where(loss_pk[:, -1] == loss_pk_min)[-1]
indx_rd = np.where(loss_rd[:, -1] == loss_rd_min)[-1]
indx_wt = np.where(loss_wt[:, -1] == loss_wt_min)[-1]

param_pk = pd.read_csv('.../COT-Flow/experiments/cnf/tabjoint/parkinson_params_hist.csv').to_numpy()
param_rd = pd.read_csv('.../COT-Flow/experiments/cnf/tabjoint/rd_wine_params_hist.csv').to_numpy()
param_wt = pd.read_csv('.../COT-Flow/experiments/cnf/tabjoint/wt_wine_params_hist.csv').to_numpy()
param_pk_list = param_pk[indx_pk]
param_rd_list = param_rd[indx_rd]
param_wt_list = param_wt[indx_wt]


for i in range(10):
    for j in range(5):
        width_pk = int(param_pk_list[i, 4])
        width_rd = int(param_rd_list[i, 4])
        width_wt = int(param_wt_list[i, 4])

        batch_size_pk = int(param_pk_list[i, -1])
        batch_size_rd = int(param_rd_list[i, -1])
        batch_size_wt = int(param_wt_list[i, -1])

        lr_pk = param_pk_list[i, 5]
        lr_rd = param_rd_list[i, 5]
        lr_wt = param_wt_list[i, 5]

        nt_pk = int(param_pk_list[i, 3])
        nt_rd = int(param_rd_list[i, 3])
        nt_wt = int(param_wt_list[i, 3])

        alpha_pk = [1.0, param_pk_list[i, 1], param_pk_list[i, 2]]
        alpha_rd = [1.0, param_rd_list[i, 1], param_rd_list[i, 2]]
        alpha_wt = [1.0, param_wt_list[i, 1], param_wt_list[i, 2]]

        os.system(
            "python trainTabularOTflowBlock.py --data 'parkinson' --dx 8 --num_epochs 1000 --drop_freq 0 \
            --val_freq 20 --weight_decay 0.0 --prec single --early_stopping 10 --lr_drop 2.0 --batch_size " + \
            str(batch_size_pk) + " --test_batch_size " + str(batch_size_pk) + " --lr " + str(lr_pk) + " --nt " + str(nt_pk) + \
            " --nt_val " + str(32) + " --m " + str(width_pk) + " --alph " + str(alpha_pk[0]) + ',' + str(alpha_pk[1]) + ',' + \
            str(alpha_pk[2]) + " --save 'experiments/cnf/tabjoint/parkinson'"
        )

        os.system(
            "python trainTabularOTflowBlock.py --data 'wt_wine' --dx 6 --num_epochs 1000 --drop_freq 0 \
            --val_freq 20 --weight_decay 0.0 --prec single --early_stopping 10 --lr_drop 2.0 --batch_size " + \
            str(batch_size_rd) + " --test_batch_size " + str(batch_size_rd) + " --lr " + str(lr_rd) + " --nt " + str(nt_rd) + \
            " --nt_val " + str(32) + " --m " + str(width_rd) + " --alph " + str(alpha_rd[0]) + ',' + str(alpha_rd[1]) + ',' + \
            str(alpha_rd[2]) + " --save 'experiments/cnf/tabjoint/white'"
        )

        os.system(
            "python trainTabularOTflowBlock.py --data 'rd_wine' --dx 6 --num_epochs 1000 --drop_freq 0 \
            --val_freq 20 --weight_decay 0.0 --prec single --early_stopping 10 --lr_drop 2.0 --batch_size " + \
            str(batch_size_wt) + " --test_batch_size " + str(batch_size_wt) + " --lr " + str(lr_wt) + " --nt " + str(nt_wt) + \
            " --nt_val " + str(32) + " --m " + str(width_wt) + " --alph " + str(alpha_wt[0]) + ',' + str(alpha_wt[1]) + ',' + \
            str(alpha_wt[2]) + " --save 'experiments/cnf/tabjoint/red'"
        )



