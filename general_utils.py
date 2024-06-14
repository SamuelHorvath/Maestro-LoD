import os
import json

import torch


def create_experiment_dir(args):
    run_id = f'id={args.identifier}'
    dir_name = args.arch

    if getattr(args, 'decomposition', None):
        dir_name += '_dec'
        if getattr(args, 'gp', None):
            dir_name += f'_gp_{str(args.gp_lambda)}'
        if getattr(args, 'progressive', None):
            dir_name += f'_prog_{str(args.importance_threshold)}'

    # to be consistent with the previous version of arguments
    if getattr(args, 'decomposition', None):
        if getattr(args, 'no_full_pass', None):
            dir_name += '_no_full'
        if getattr(args, 'full_training_epochs', None):
            dir_name += f'_full_{str(args.full_training_epochs)}'
        if getattr(args, 'ignore_k_first_layers', None):
            dir_name += f'_ignore_k_{str(args.ignore_k_first_layers)}'
        if getattr(args, 'ignore_last_layer', None):
            dir_name += '_ignore_last'

    dir_name += f'_lr_{str(args.lr)}'
    dir_name += f'_seed_{str(args.seed)}'

    experiment_dir = os.path.join(
        args.outputs_dir, run_id, dir_name
    )
    return experiment_dir


def get_exp_run(args, load_models=False):

    experiment_dir = create_experiment_dir(args)
    train_metrics_dir = os.path.join(
        experiment_dir, 'full_metrics_train.json')
    test_metrics_dir = os.path.join(
        experiment_dir, 'full_metrics_test.json')
    importances_dir = os.path.join(
        experiment_dir, 'importances.json')
    # finished_dir = os.path.join(
    #     experiment_dir, 'finished.json')
    model_last_dir = os.path.join(
        experiment_dir, 'last_model.pt')
    model_best_dir = os.path.join(
        experiment_dir, 'best_model.pt')

    # if not os.path.exists(finished_dir):
    #     print(f'Experiment does not exists {finished_dir}.')
    #     return (None, ) * 5

    with open(test_metrics_dir, 'r') as f:
        test_dict = json.load(f)
    with open(train_metrics_dir, 'r') as f:
        train_dict = json.load(f)
    with open(importances_dir, 'r') as f:
        importances_dict = json.load(f)
    if not load_models:
        return test_dict, train_dict, importances_dict, None, None
    models_last = torch.load(model_last_dir)
    models_best = torch.load(model_best_dir)
    return test_dict, train_dict, importances_dict, \
        models_best, models_last


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def checkpoint_model(filename, model, optimizer, epoch, test_loss, best_loss):
    if (best_loss and test_loss <= best_loss) or best_loss is None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                if optimizer else dict(),
                "loss": test_loss,
            }, filename
        )
