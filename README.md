# Backpropagation-Friendly Eigendecomposition
Eigendecomposition (ED) is widely used in deep networks. However, the backpropagation of its results tends to be numerically unstable, whether using ED directly or approximating it with the Power Iteration method, particularly when dealing with large matrices. While this can be mitigated by partitioning the data in small and arbitrary groups, doing so has no theoretical basis and makes its impossible to exploit the power of ED to the full. We introduce a numerically stable and differentiable approach to leveraging eigenvectors in deep networks. It can handle large symmetric square matrices without requiring to split them.
## Github Code
## Pros & cons
Pros:
- Numerically Stable.
- Can be plugged into any codes easily.

Cons:
- Could not compute all eigenvalues (for very large matrices) because of the round-off error accumulation.
- It is a bit slow as it has for loops inside the deflation process.
- SVD forward pass is very slow for matricies whose size is larger than 128.

ZCA layer is nested at the bottom of the first residual block of ResNet.
The 64 channels are partitioned into N groups and each group has d=64/N channels.
The covariance matrix is computed for each group with the dimension of d.


## Accuracy on Cifar10
| Model             |Dim. (d)| Min Acc.    | Mean Acc.  |
| ----------------- | ------ | ----------- |----------- |
| ResNet18          | -      | 93.02%      | -     |
| ResNet18+ZCA      | 4      | 95.41%      | 95.29±0.11%     |
| ResNet18+ZCA      | 8      | 95.57%      | 95.38±0.18%     |
| ResNet18+ZCA      | 16     | 95.60%      | 95.37±0.14%     |
| ResNet18+ZCA      | 32     | 95.54%      | 95.36±0.15%     |
| ResNet18+ZCA      | 64     | 95.56%      | 95.41±0.09%     |

## Learning rate adjustment
the `lr` is changed as follows:
- `0.1` for epoch `[0,100)`
- `0.01` for epoch `[100,200)`
- `0.001` for epoch `[200,300)`
- `0.0001` for epoch `[300,350)`

##
The pytorch must be GPU version, as we have not test the code on CPU machine.
Now our code does not support multi-GPU setting.
You need to run the following command to train the model.
Here are the code for training ResNet18 on CIFAR10 & CIFAR100

## training
On CIFAR10 Dataset: \
run ZCA whitening: `python main.py --norm=zcanormsvdpi` \
run PCA denpoising: `python main.py --norm=pcanormsvdpi`

On CIFAR100 Dataset: \
run ZCA whitening: `python main_cifar100.py --norm=zcanormsvdpi` \
run PCA denpoising: `python main_cifar100.py --norm=pcanormsvdpi`

## requirement
The code might not be compatible with lower version of the specified packages.

```
Python = 3.7.2
PyTorch >= 1.1.0
Torchvision >= 0.2.2
Scipy >= 1.2.1
Numpy >= 1.16.3
tensorboardX
```

## Paper
```
@inproceedings{wang2019backpropagation,
  title={Backpropagation-Friendly Eigendecomposition},
  author={Wang, Wei and Dang, Zheng and Hu, Yinlin and Fua, Pascal and Salzmann, Mathieu},
  booktitle={Advances in neural information processing systems},
  year={2019}
}
```
## Acknowledgement
The code is heavily based on [pytorch-cifar][https://github.com/kuangliu/pytorch-cifar]
