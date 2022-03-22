this code is based on 
- [SALIENCYMIX](https://github.com/afm-shahab-uddin/SaliencyMix)

-To train PYTORCH-BASE(naive ver.):    
```
CUDA_VISIBLE_DEVICES=0 python train.py --net_type resnet --dataset cifar100 --depth 18 --alpha 240 --batch_size 128 --lr 0.25 --expname PyraNet200 --epochs 2 --beta 1.0 --cutmix_prob 0.5
```

-To train PYTORCH-AMP:    
```
CUDA_VISIBLE_DEVICES=0 python trainAmp.py --net_type resnet --dataset cifar100 --depth 18 --alpha 240 --batch_size 128 --lr 0.25 --expname PyraNet200 --epochs 2 --beta 1.0 --cutmix_prob 0.5
```

-To train PYTORCH-DDP:    
```
python train-multi-ddp.py --net_type pyramidnet --dataset cifar100 --depth 200 --alpha 240 --batch_size 512 --lr 0.25 --expname PyraNet200 --epochs 1 --beta 1.0 --cutmix_prob 0.5 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
```
-To train PYTORCH-DDP-AMP:    
```
python train-multi-ddp-amp.py --net_type pyramidnet --dataset cifar100 --depth 200 --alpha 240 --batch_size 512 --lr 0.25 --expname PyraNet200 --epochs 1 --beta 1.0 --cutmix_prob 0.5 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
```
