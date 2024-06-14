import argparse
import time

from types import SimpleNamespace

from maestro.models import resnets as models


def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parse_maestro_opts(parser):
    parser.add_argument('--decomposition', action='store_true',
                        default=False,
                        help='use layer decomposition')
    parser.add_argument('--progressive', action='store_true',
                        default=False,
                        help='use progressive pruning for decomposition')

    parser.add_argument('--gp', action='store_true',
                        default=False,
                        help='use Group Lasso penalty')
    parser.add_argument('--gp-lambda', default=5e-3, type=float,
                        help='Group Lasso regulariser multiplier')
    parser.add_argument('--importance-threshold', default=1e-5, type=float,
                        help='Importance threshold for zeroing out values')
    parser.add_argument('--od-sampler', default=None,
                        choices=['across_layers', 'per_layer', 'pufferfish'],
                        help='Type of sampler to use.')
    parser.add_argument("--no-full-pass", action="store_true", default=False,
                        help="Whether not to do full pass of the network")
    # Pufferfish Setup
    parser.add_argument("--full-training-epochs", default=0, type=int,
                        help="Number of epochs for full training")
    parser.add_argument("--ignore-k-first-layers", default=0, type=int,
                        help="Number of layers to ignore from decomposition")
    parser.add_argument("--ignore-last-layer", action="store_true",
                        default=False,
                        help="Whether to ignore the last layer from "
                        "decomposition")

    # experiment directory
    parser.add_argument("--outputs-dir", type=str,
                        default="./outputs/",
                        help="Base root directory for the output.")
    parser.add_argument("--identifier", type=str, default=str(time.time()),
                        help="Identifier for the current job")

    return parser


# ImageNet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_imagenet_opts(args):
    parser = initialise_arg_parser(
        args, description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256),'
                        ' this is the total'
                        ' batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training'
                        'to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or'
                        ' multi node data parallel training')
    parser.add_argument('--dummy', action='store_true',
                        help="use fake data to benchmark")
    return parser


def parse_cifar_opts(args):
    parser = initialise_arg_parser(args, 'Maestro CIFAR/MNIST implementation')
    # generic arguments
    parser.add_argument('--evaluate', type=bool_string, default=False,
                        help='if or not to evaluate the save model and exit.')
    parser.add_argument('--evaluate-from', type=str, default='./best_model.pt',
                        help='path to evaluate model from')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='how many epochs to wait before evaluating '
                             'test performance')

    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    parser.add_argument("--outputs-dir", type=str,
                        default="./outputs/",
                        help="Base root directory for the output.")
    parser.add_argument("--identifier", type=str, default=str(time.time()),
                        help="Identifier for the current job")

    # model specific arguments
    parser.add_argument('--model', type=str, help="model string")
    parser.add_argument('--batch-norm', action='store_true',
                        help='use BatchNorm2d as norm layer (else GroupNorm)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--eval-batch-size', type=int, default=128,
                        metavar='N',
                        help='evaluation batch size')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                        help='optimiser momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='N', help='optimiser momentum')

    return parser


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(
        args, description=description, allow_abbrev=False)
    return parser


def merge_args(args1, args2):
    new_dict = vars(args1).copy()
    new_dict.update(vars(args2))

    new_args = SimpleNamespace(**new_dict)

    return new_args


def parse_args(args, experiment_type='cifar'):
    maestro_parser = initialise_arg_parser(args, "Maestro Layers")
    parse_maestro_opts(maestro_parser)
    maestro_args, unparsed_args = maestro_parser.parse_known_args()

    if experiment_type == 'imagenet':
        parser = parse_imagenet_opts(unparsed_args)
    elif experiment_type == 'cifar_mnist':
        parser = parse_cifar_opts(unparsed_args)
    else:
        raise ValueError("Unknown experiment type")

    args = parser.parse_known_args()[0]
    args = merge_args(maestro_args, args)

    print(args)
    # validate_args(args)

    return args
