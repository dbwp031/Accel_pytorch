# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np

import torch.multiprocessing as mp
import builtins
import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

def main(): 
    args = parser.parse_args()

    if args.dist_url=='env://' and args.world_size==-1:
        args.world_size=int(os.environ['WORLD_SIZE']) # 현재는 에러뜸
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size=ngpus_per_node*args.world_size
        mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    else:
        main_worker(args.gpu,ngpus_per_node,args)    


def main_worker(gpu,ngpus_per_node, args):
    # 내용1 :gpu 설정
    args.gpu = gpu

    global best_err1, best_err5

    if args.multiprocessing_distributed and args.gpu !=0:
        def print_pass(*args):
            pass
        builtins.print=print_pass
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url=='env://' and args.rank==-1:
            args.rank=int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank=args.rank*ngpus_per_node + gpu #gpu None아님?
        torch.distributed.init_process_group(backend=args.dist_backend,init_method=args.dist_url,
                                            world_size=args.world_size,rank=args.rank)
    # 내용2: model 정의
    print("=> creating model '{}'".format(args.net_type))
    if args.dataset == 'cifar100':
        numberofclass = 100
    elif args.dataset == 'cifar10':
        numberofclass = 10
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))
    # 내용3: multiprocess 설정
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # when using a single GPU per process and per DDP, we need to divide tha batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers+ngpus_per_node-1)/ngpus_per_node)
            # 내용3-1: model ddp설정
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])# args.gpu가 무슨 값인지 알고 싶다.
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
            # 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model=model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataparallel is supported.")

    # 내용4: criterion / optimizer 정의
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # 옵션1: resume 방법
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint: '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # MAP model to be loaded to specific single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict((checkpoint['state_dict']))
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True # 얘는 뭐하는 거지?
    
    # 내용5: 데이터 로딩
    # 내용5-1: transform 정의
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        # 내용5-2: dataset 정의
        if args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR100('../data', train=False, transform=transform_test)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10('../data', train=False, transform=transform_test)
            numberofclass = 10
            
        # 내용5-3: sampler 정의 (참고: val_loader는 sampler를 사용하지 않는다.)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True,sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    # 내용 6: for문을 통한 training
    cudnn.benchmark = True
    stime = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 내용 6-1: train_sampler.set_epoch
        # In distributed mode, calling the set_eopch() method at the beggining of 
        # each epoch before creating the "dataloader" iterator is necessary to make
        # suffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch,args)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch,args)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch,args)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank% ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
                'optimizer': optimizer.state_dict(),
            }, is_best,args=args)
    etime = time.time()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        print("총 걸린시간: ",etime-stime)
        print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch,args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch,args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input.cuda())
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best,args, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch,args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
