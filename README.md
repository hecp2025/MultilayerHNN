# MultilayerHNN

This is the code for adaptive learning for memristor associative memory and multilayer structure. This is the source code for the paper: Hardware-Adaptive and Superlinear-Capacity Memristor-based Associative Memory.

The run command for training is:
```python
python multilayer.py --dimension 784 --interval 1 --train-eval 'train' --variation '0.0' --stuck '0.0' --corruption 0.05 -seed 1 --dataset 'mnist' --binary True --max-pattern 64 --min-pattern 1
```

The run command for evaluate is:
```python
python multilayer.py --dimension 784 --interval 1 --train-eval 'eval' --variation '0.0' --stuck '0.0' --corruption 0.05 -seed 1 --dataset 'mnist' --binary True --max-pattern 64 --min-pattern 1
```
