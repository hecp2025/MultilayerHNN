# MultilayerHNN

This is the code for adaptive learning for memristor associative memory and multilayer structure. This is the source code for the paper: Hardware-Adaptive and Superlinear-Capacity Memristor-based Associative Memory.

The run command for different methods is:
```python
python --dimension 784 --interval 1 --train-eval 'train' --varaition '0.0' --stuck '0.0' --corruption 0.05 -seed 1 --dataset 'mnist' --binary True --max-pattern 64 --min-pattern 1
```
