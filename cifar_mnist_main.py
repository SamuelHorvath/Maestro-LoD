from __future__ import print_function

import sys
import os
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from maestro_opts import parse_args

from general_utils import checkpoint_model
from maestro.layers.utils import bn_calibration_init, \
    group_lasso_criterion, progressive_shrinking
from maestro.samplers.utils import get_sampler
from maestro.layers.decomposition import decompose_model

from general_utils import \
    create_experiment_dir


DATASETS = {
    'resnet18': 'cifar10',
    'vgg19': 'cifar10',
    'lenet': 'mnist',
}


GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}


def create_norm_layer(num_channels, batch_norm=True):
    if batch_norm:
        return nn.BatchNorm2d(num_channels)
    return nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)


def get_dataset(dataset_name, data_dir='./data'):
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(
            root=data_dir, train=True, download=True,
            transform=transform_train)
        testset = datasets.CIFAR10(
            root=data_dir, train=False, download=True,
            transform=transform_test)
    elif dataset_name == 'mnist':
        trainset = datasets.MNIST(
            data_dir, train=True, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]))

        testset = datasets.MNIST(
            data_dir, train=False, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]))
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    return trainset, testset


def initialise_model(model_str, device, args):
    if model_str == 'resnet18':
        from maestro.models.resnets import resnet18

        def norm_layer(num_channels):
            return create_norm_layer(num_channels, args.batch_norm)

        model = resnet18(
            norm_layer=norm_layer)
        input_size = (3, 32, 32)
    elif model_str == 'vgg19':
        from maestro.models.vggs import VGG

        def norm_layer(num_channels):
            return create_norm_layer(num_channels, args.batch_norm)

        model = VGG(
            'VGG19',
            norm_layer=norm_layer)
        input_size = (3, 32, 32)
    elif model_str == 'lenet':
        from maestro.models.lenet import MaestroLeNet
        model = MaestroLeNet()
        input_size = (1, 28, 28)

    else:
        raise ValueError(f"Unknown model {model_str}")

    return model.to(device), input_size


def get_scheduler(network_name, optimizer, args):
    if network_name in ['resnet18', 'vgg19']:
        scheduler = MultiStepLR(optimizer, [int(0.5 * args.epochs),
                                int(0.75 * args.epochs)], gamma=0.1)
    elif network_name == 'lenet':
        scheduler = MultiStepLR(optimizer, [args.epochs + 1], gamma=0.1)
    else:
        raise NotImplementedError(
            f"Scheduler for {network_name} not implemented.")
    return scheduler


def train(args, model, device, train_loader, optimizer, epoch,
          criterion, od_sampler, hierarchical):
    model.train()
    train_loss = 0
    data_processed = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if args.no_full_pass:
            loss_full = 0
        else:
            output_full = model(data)
            loss_full = criterion(output_full, target)
            target = output_full.detach().softmax(dim=1)

        loss_partial = 0
        if args.decomposition:
            output_partial = model(data, sampler=od_sampler)
            loss_partial = criterion(output_partial, target)

        group_lasso_loss = 0
        if args.gp:
            group_lasso_loss = group_lasso_criterion(
                model, hierarchical=hierarchical)

        # loss_total = (loss_full + loss_partial) / n_losses
        loss_total = (loss_full + loss_partial)
        loss = loss_total + args.gp_lambda * group_lasso_loss

        loss.backward()
        optimizer.step()

        batch_size = data.shape[0]
        train_loss += loss.item() * batch_size
        data_processed += batch_size

        if batch_idx % args.log_interval == 0 or \
           batch_idx == len(train_loader) - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, data_processed, len(train_loader.dataset),
                100. * data_processed / len(train_loader.dataset),
                train_loss / data_processed))
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item() * batch_size
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f},'
          'Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))

    return test_loss, test_acc


def main():
    # Training settings
    args = parse_args(sys.argv, 'cifar_mnist')
    args.arch = args.model

    use_cuda = torch.cuda.is_available()

    # DATASETS
    dataset = DATASETS[args.model]
    trainset, testset = get_dataset(dataset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model, _ = initialise_model(
        args.model, device, args)

    # decompose the model if needed and test sampler
    if args.decomposition:
        decompose_model(
            model=model
            )

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_scheduler(args.model, optimizer, args)

    od_sampler = get_sampler(args.od_sampler, model, with_layer=False)
    # whether to use hierarchical pruning
    hierarchical = True
    print(f"Using hierarchical pruning: {hierarchical}")

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

    best_acc = -float('inf')
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
    for m in model.modules():
        if hasattr(m, 'inner_dim'):
            i += 1
            importances_dict[f'{m._get_name()}_{i}'] = []

    for epoch in range(1, args.epochs + 1):
        for group in optimizer.param_groups:
            print("### Epoch: {}, Current effective lr: {}".format(
                epoch, group['lr']))
            break

        train_loss = train(args, model, device, train_loader, optimizer, epoch,
                           criterion, od_sampler, hierarchical=hierarchical)
        scheduler.step()

        train_dict['epoch'].append(epoch)
        train_dict['train_loss'].append(train_loss)
        with torch.no_grad():
            i = 0
            for m in model.modules():
                if hasattr(m, 'inner_dim'):
                    i += 1
                    importances_dict[f'{m._get_name()}_{i}'].append(
                        m.importance(hierarchical).cpu().numpy().tolist())

        if args.progressive:
            print("Progressive shrinking ...")
            progressive_shrinking(
                model, args.importance_threshold, hierarchical=hierarchical)
            if od_sampler is not None:
                od_sampler.prepare_sampler()

        if epoch % args.eval_interval == 1 or epoch == args.epochs:
            if args.batch_norm and args.decomposition:
                # reset batch_norm statistics to the current model and data
                for m in model.modules():
                    bn_calibration_init(m)

                model.train()
                for data, _ in train_loader:
                    data = data.to(device)
                    model(data)

            test_loss, test_acc = test(model, device, test_loader, criterion)

            test_dict['epoch'].append(epoch)
            test_dict['test_loss'].append(test_loss)
            test_dict['test_acc'].append(test_acc)

            # test metrics
            checkpoint_model(
                best_model_dir, model, optimizer, epoch=epoch,
                test_loss=-test_acc, best_loss=-best_acc)
            with open(test_metrics_dir, 'w') as f:
                json.dump(test_dict, f, indent=4)
            if test_acc >= best_acc:
                best_acc = test_acc

        # train metrics
        checkpoint_model(
            last_model_dir, model, optimizer, epoch=epoch,
            test_loss=0., best_loss=1.)
        with open(train_metrics_dir, 'w') as f:
            json.dump(train_dict, f, indent=4)
        with open(importances_dir, 'w') as f:
            json.dump(importances_dict, f, indent=4)
    with open(finished_dir, 'w') as f:
        json.dump({}, f, indent=4)


if __name__ == '__main__':
    main()
