# COT-Flow
Pytorch implementation of the conditional version of the OT-Flow approach.

## Associated Publication

OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17113

Supplemental: https://arxiv.org/abs/2006.00104

Please cite as
    
    @inproceedings{onken2021otflow, 
        title={{OT-Flow}: Fast and Accurate Continuous Normalizing Flows via Optimal Transport},
        author={Derek Onken and Samy Wu Fung and Xingjian Li and Lars Ruthotto},
	    volume={35}, 
	    number={10}, 
	    booktitle={AAAI Conference on Artificial Intelligence}, 
	    year={2021}, 
	    month={May},
	    pages={9223--9232},
	    url={https://ojs.aaai.org/index.php/AAAI/article/view/17113}, 
    }

Efficient Neural Network Approaches for Conditional Optimal Transport with 
applications in Bayesian Inference.

Paper: https://arxiv.org/abs/2310.16975

## Set-up

To run some files (e.g. the tests), you may need to add them to the path via
```
export PYTHONPATH="${PYTHONPATH}:."
```

## UCI Tabular Datasets Experiments

Due to copyright concerns, we did not provide the required datasets. 
They can be downloaded from the UC Irvine Machine Learning Repository. 

After downloading, please specify the paths to these 
datasets in file 'datasets/tabular_data.py' for loading purposes.

#### Perform pilot runs to search for best hyperparameter combinations:

```
python pretrainTabOTflowBlock.py --data 'parkinson' --dx 8
python pretrainTabOTflowBlock.py --data 'rd_wine' --dx 6
python pretrainTabOTflowBlock.py --data 'wt_wine' --dx 6

python pretrainOTflowCond.py --data 'concrete'
python pretrainOTflowCond.py --data 'energy'
python pretrainOTflowCond.py --data 'yacht'
```

#### Perform experiments with the 10 best hyperparameter combinations from pilot runs:
Before running following scripts, please change accordingly to correct file paths 
when loading datasets and hyperparameter combinations
```
python experimentBlock.py
python experimentCond.py
```

#### Evaluate the trained model
```
python evaluateTabularOTflowCond.py
python evaluateTabularOTflowBlock.py
```

## Stochastic Lotka-Volterra Experiment

#### Prepare training dataset:
Before running this script, change to correct absolute path for storing the dataset
```
python sample_stoch_lv.py
```

#### Perform pilot runs to search for best hyperparameter combination:
```
python pretrainOTflowCond.py --data 'lv' --dx 4
```

#### Perform training with the best hyperparameter combination:
Before running following script, please change accordingly to correct file paths 
when loading datasets and hyperparameter combinations
```
python experimentLV.py
```

#### Evaluate the trained model
Before running following script, please change accordingly to correct file paths 
```
python evaluateLV.py
```

## 1D Shallow Water Equations Experiment

#### Prepare training dataset:

Please change the "path_to_fcode" variable in "simulator.py" to the correct
absolute path to "shallow_water01_modified.f90".

Change the "--path_to_save" argument in "sample_shallow_water.py" to correct paths

```
for k in 1 2 3 4 5 6 7 8 9 10
do
  python sample_shallow_water.py --job_num $k
done
```

#### Process dataset (dimension reduction for observations 'y'):
Before running following script, please change accordingly to correct file paths 
when loading datasets
```
python shallow_water.py
```

The datasets used for the associated paper can be found through
https://drive.google.com/drive/folders/1ObuuATIEsC3z9d0S_WRp2lGZVOClu0Ip?usp=drive_link. Note
that the datasets here contain already projected, 3500-dimensional, observations "y". 

If using the datasets under the Google Drive link above, please make the following modification:

Change code piece "Vs = torch.FloatTensor(datafile['Vs'])" (line 72 of 'shallow_water.py')
to "Vs = torch.FloatTensor(datafile['V'])"


#### Perform pilot runs to search for best hyperparameter combination:
```
python pretrainOTflowSW
```

#### Perform training with the best hyperparameter combination:
Before running following script, please change accordingly to correct file paths 
when loading hyperparameter combinations
```
python experimentSW.py
```

#### Evaluate the trained model
Before running following script, please change accordingly to correct file paths 
```
python evaluateSW.py
```







