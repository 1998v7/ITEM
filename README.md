# ITEM
Pytorch implementation of the paper "Debiased Sample Selection for Combating Noisy Labels"

### Run

For **CIFAR-10/100** with symmetric or instance-dependent label noise
```
python Train_cifar.py --dataset ['cifar10', 'cifar100']
                      --batch_size 64
                      --noise_mode sym
                      --r 0.2
                      --cls_num 4
                      --beta 3
                      --gpuid 0
```

For **CIAFAR-10N** with worst, random 1/2/3.
```
python Train_cifarN.py --noise_mode ['worse_label', 'random_label1', 'random_label2', 'random_label3']
                       --cls_num 4
                       --beta 3
                       --gpuid 0
```
