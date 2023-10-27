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
Perform pilot runs to search for best hyperparameter combinations:

```
python pretrainTabOTflowBlock.py --data 'parkinson' --dx 8
python pretrainTabOTflowBlock.py --data 'rd_wine' --dx 6
python pretrainTabOTflowBlock.py --data 'wt_wine' --dx 6

python pretrainOTflowCond.py --data 'concrete'
python pretrainOTflowCond.py --data 'energy'
python pretrainOTflowCond.py --data 'yacht'
```

Perform experiments with the 10 best hyperparameter combinations from pilot runs:
```
python experimentBlock.py
python experimentCond.py
```

Evaluate the trained model
```
python evaluateTabularOTflowCond.py
python evaluateTabularOTflowBlock.py
```

## Stochastic Lotka-Volterra Experiment

Perform pilot runs to search for best hyperparameter combination:
```
python pretrainOTflowCond.py --data 'lv' --dx 4
```

Perform training with the best hyperparameter combination:
```
python experimentLV.py
```

Evaluate the trained model
```
python evaluateLV.py
```

## 1D Shallow Water Equations Experiment
Perform pilot runs to search for best hyperparameter combination:
```
python pretrainOTflowSW
```

Perform training with the best hyperparameter combination:
```
python experimentSW.py
```

Evaluate the trained model
```
python evaluateSW.py
```







