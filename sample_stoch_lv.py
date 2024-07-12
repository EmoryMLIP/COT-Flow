import scipy.io
from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra

num_samples = 50000
stoch = StochasticLotkaVolterra()
x_train, y_train = stoch.sample_joint(num_samples)

# TODO specify file path
file_name = '.../COT-Flow/datasets/lv_data.mat'
scipy.io.savemat(file_name, {'x_train': x_train, 'y_train': y_train})
