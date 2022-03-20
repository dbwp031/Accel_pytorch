this code is based on 
- [SALIENCYMIX](https://github.com/afm-shahab-uddin/SaliencyMix)

-To train ResNet50 on CIFAR10 with SaliencyMix and traditional data augmentation:    
```
CUDA_VISIBLE_DEVICES=0 python trainAmp.py --net_type resnet --dataset cifar100 --depth 18 --alpha 240 --batch_size 128 --lr 0.25 --expname PyraNet200 --epochs 2 --beta 1.0 --cutmix_prob 0.5
```
