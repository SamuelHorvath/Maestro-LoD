# adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py

import os
import sys
import random
import shutil
import json
import time
import warnings
from enum import Enum
# import wandb
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

from maestro_opts import parse_args
from maestro.layers.utils import bn_calibration_init, \
    group_lasso_criterion, progressive_shrinking
from maestro.layers.decomposition import decompose_model
from maestro.samplers.utils import get_sampler
from maestro.models import resnets as models

from general_utils import create_experiment_dir, \
    get_exp_run, add_weight_decay


best_acc1 = 0


def main():
    args = parse_args(sys.argv, "imagenet")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # create experiment directory
    experiment_dir = create_experiment_dir(args)
    train_metrics_dir = os.path.join(
        experiment_dir, 'full_metrics_train.json')
    test_metrics_dir = os.path.join(
        experiment_dir, 'full_metrics_test.json')
    importances_dir = os.path.join(
        experiment_dir, 'importances.json')
    finished_dir = os.path.join(
        experiment_dir, 'finished.json')

    if os.path.exists(finished_dir):
        print(f"{experiment_dir} already exists.")
        return
    os.makedirs(experiment_dir, exist_ok=True)

    best_model_dir = os.path.join(
        experiment_dir, 'best_model.pt')
    last_model_dir = os.path.join(
        experiment_dir, 'last_model.pt')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model_decomposed = deepcopy(model)

    # decompose the model if needed and test sampler
    if args.decomposition:
        decompose_model(
            model=model_decomposed,
            ignore_k_first_layers=args.ignore_k_first_layers,
            ignore_last_layer=args.ignore_last_layer
            )

    # whether to use hierarchical pruning, default True
    hierarchical_reg = True
    print(f"Using hierarchical pruning: {hierarchical_reg}")

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                model_decomposed.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of
                # GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) /
                                   ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
                model_decomposed = torch.nn.parallel.DistributedDataParallel(
                    model_decomposed, device_ids=[args.gpu])
            else:
                model.cuda()
                model_decomposed.cuda()
                # DistributedDataParallel will divide and allocate
                # batch_size to all available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
                model_decomposed = torch.nn.parallel.DistributedDataParallel(
                    model_decomposed)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_decomposed = model_decomposed.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        model_decomposed = model_decomposed.to(device)
    else:
        # DataParallel will divide and allocate batch_size
        # to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            model_decomposed.features = torch.nn.DataParallel(
                model_decomposed.features)
            model_decomposed.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            model_decomposed = torch.nn.DataParallel(
                model_decomposed).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    parameters = add_weight_decay(model, args.weight_decay)
    # weight decay applied above
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=0.)

    """Sets the learning rate to the initial
        LR decayed by 10 in 30, 60, 80 epochs"""
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    train_dict = {
        'epoch': [],
        'train_loss': []
    }
    test_dict = {
        'epoch': [],
        'test_loss': [],
        'test_acc': []
    }

    importances_dict = {}
    i = 0
    for m in model_decomposed.modules():
        if hasattr(m, 'inner_dim'):
            i += 1
            importances_dict[f'{m._get_name()}_{i}'] = []

    # Resume from a checkpoint if you already
    # run part of the experiment
    args.resume = last_model_dir
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # switch to decomposition if already in that phase
            if args.start_epoch > args.full_training_epochs:
                # decompose model
                od_sampler, optimizer, scheduler = \
                    switch_to_decomposition(
                        model, model_decomposed,
                        args, args.start_epoch, load_model=False)
                model_decomposed.load_state_dict(checkpoint['state_dict'])
            else:
                od_sampler = None
                model.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and isinstance(best_acc1, torch.Tensor):
                # best_acc1 may be from a checkpoint from a different GPU
                if isinstance(best_acc1, torch.Tensor):
                    best_acc1 = best_acc1.to(args.gpu)
            # od_sampler = get_sampler(
            #     args.od_sampler, model, with_layer=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            test_dict, train_dict, importances_dict, _, _ = get_exp_run(
                args, load_models=False)
            print(test_dict, train_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        my_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Loaded model at epoch {args.start_epoch}, lr {my_lr}")

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(
            256, (3, 224, 224), 1000, transforms.ToTensor())  # 1281167
        val_dataset = datasets.FakeData(
            256, (3, 224, 224), 1000, transforms.ToTensor())  # 50000
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # wandb
    # if not args.multiprocessing_distributed or \
    #         (args.multiprocessing_distributed and
    #          args.rank == 0):
    #     run_id = '_'.join(experiment_dir.split('/')[-2:])
    #     wandb.init(
    #         ...
    #     )

    args_training = deepcopy(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch < args.full_training_epochs:
            # Full training params
            args_training.no_full_pass = False
            args_training.decomposition = False
            args_training.progressive = False
            args_training.gp = False
            args_training.gp_lambda = 0
            od_sampler = None
            model_training = model
            print("=> Full training...")

        elif args.full_training_epochs == epoch:
            print("Switching to factorized training...")
            # back to original params
            args_training = deepcopy(args)
            # decompose model
            od_sampler, optimizer, scheduler = switch_to_decomposition(
                model, model_decomposed, args, epoch)
            model_training = model_decomposed

            # check that OD sampler works correctly
            if args.decomposition:
                dummy_sampler = get_sampler(
                    args.od_sampler, model, with_layer=True)
                images, _ = next(iter(train_loader))
                images = images.to(device)
                print("=> Testing OD sampler...")
                model(images, dummy_sampler)
                print("=> OD sampler works correctly!")
        else:
            model_training = model_decomposed

        # train for one epoch
        print("Training the model...")
        train_loss = train(
            train_loader, model_training, criterion, optimizer, epoch, device,
            args_training, od_sampler, hierarchical_reg)

        train_dict['epoch'].append(epoch)
        train_dict['train_loss'].append(train_loss)

        print("Computing importances...")
        with torch.no_grad():
            i = 0
            for m in model_decomposed.modules():
                if hasattr(m, 'inner_dim'):
                    i += 1
                    importances_dict[f'{m._get_name()}_{i}'].append(
                        m.importance(hierarchical_reg).cpu().numpy().tolist())

        scheduler.step()

        # validation_sampler, only for pufferfish in decomposition stage,
        # we use sampler to sample the rank
        if args.od_sampler == 'pufferfish' and args_training.decomposition:
            val_od_sampler = get_sampler(
                args.od_sampler, model_decomposed, with_layer=False)
            print("Pufferfish sampler is used for validation.")
        else:
            val_od_sampler = None

        # progressively shrink the network
        if args_training.progressive:
            print("Progressively shrinking the network...")
            progressive_shrinking(
                model_decomposed, args.importance_threshold,
                hierarchical=hierarchical_reg)
            # adapt to new ranks
            if od_sampler is not None:
                od_sampler.prepare_sampler()

        # recompute batch norm statistics
        if args_training.decomposition:
            print("Recomputing batch norm statistics...")
            # check there is batch norm layer in the model
            if any(isinstance(m, nn.BatchNorm2d)
                   for m in model_decomposed.modules()):
                for m in model_decomposed.modules():
                    bn_calibration_init(m)
                # recompute batch norm statistics
                model_decomposed.train()
                total_batches = 0
                for images, _ in train_loader:
                    images = images.to(device, non_blocking=True)
                    model_decomposed(images)
                    total_batches += images.shape[0]
                    if total_batches >= 128000:
                        break

        # evaluate on validation set
        print("Evaluating the model...")
        test_acc1, test_loss = validate(
            val_loader, model_training, criterion, args,
            val_od_sampler=val_od_sampler)

        test_dict['epoch'].append(epoch)
        test_dict['test_loss'].append(test_loss)
        test_dict['test_acc'].append(test_acc1)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and
                 args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_training.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, last_model_dir, best_model_dir)
            with open(train_metrics_dir, 'w') as f:
                json.dump(train_dict, f, indent=4)
            with open(test_metrics_dir, 'w') as f:
                json.dump(test_dict, f, indent=4)
            with open(importances_dir, 'w') as f:
                json.dump(importances_dict, f, indent=4)
        # wandb
        # if not args.multiprocessing_distributed or \
        #         (args.multiprocessing_distributed and
        #          args.rank == 0):
        #     print("Logging to wandb...")
        #     wandb.log(
        #         {
        #             "train_loss": train_loss,
        #             "test_loss": test_loss,
        #             "test_acc": test_acc1,
        #         },
        #         step=epoch
        #     )

    # label the experiment as finished
    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and
             args.rank % ngpus_per_node == 0):
        with open(finished_dir, 'w') as f:
            json.dump({}, f, indent=4)


def switch_to_decomposition(model, model_decomposed, args, current_epoch,
                            load_model=True):
    if load_model:
        # decompose model
        if args.decomposition:
            decompose_model(
                model=model,
                ignore_k_first_layers=args.ignore_k_first_layers,
                ignore_last_layer=args.ignore_last_layer)
        model_decomposed.load_state_dict(model.state_dict())
    od_sampler = get_sampler(
        args.od_sampler, model_decomposed, with_layer=False)

    # reset optimizer and scheduler
    parameters = add_weight_decay(model_decomposed, args.weight_decay)
    # weight decay applied above
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=0.)

    """Sets the learning rate to the initial
        LR decayed by 10 in 30, 60, 80 epochs"""
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    for _ in range(current_epoch):
        scheduler.step()

    return od_sampler, optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, device, args,
          od_sampler, hierarchical_reg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute loss
        if args.no_full_pass:
            loss_full = 0
            target_partial = target
        else:
            output_full = model(images)
            loss_full = criterion(output_full, target)
            loss_full.backward()
            # do KD for sampled network
            target_partial = output_full.detach().softmax(dim=1)

        loss_partial = 0
        if args.decomposition:
            output_partial = model(images, sampler=od_sampler)
            loss_partial = criterion(output_partial, target_partial)
            loss_partial.backward()

        group_lasso_loss = 0
        if args.gp:
            group_lasso_loss = args.gp_lambda * group_lasso_criterion(
                model, hierarchical=hierarchical_reg)
            group_lasso_loss.backward()

        optimizer.step()
        loss = loss_full + loss_partial + group_lasso_loss

        if args.no_full_pass:
            if args.decomposition:
                output = output_partial
            else:
                raise ValueError("No full pass and no decomposition!")
        else:
            output = output_full

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    return losses.avg


def validate(val_loader, model, criterion, args, val_od_sampler):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images, val_od_sampler)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (
            len(val_loader.sampler) * args.world_size < len(
                val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(
            val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size,
                  len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    top1_avg = top1.avg.item() if isinstance(
        top1.avg, torch.Tensor) else top1.avg
    losses_avg = losses.avg.item() if isinstance(
        losses.avg, torch.Tensor) else losses.avg

    return top1_avg, losses_avg


def save_checkpoint(state, is_best, last_filename, best_filename):
    torch.save(state, last_filename)
    if is_best:
        shutil.copyfile(last_filename, best_filename)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
       for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
