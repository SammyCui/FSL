import numpy as np
import torch
from torch import optim
from model.protonet import ProtoNet


def sort_batch(batch):
    data, labels = batch
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    return data, labels


def partition_task(n_ways, n_shots, n_queries):
    """
    partition proto and query set for training data
    :param n_ways:
    :param n_shots:
    :param n_queries:
    :return:
    """

    support_indices = torch.arange(0, n_ways * (n_shots + n_queries), n_shots + n_queries)
    mask = torch.ones(n_ways * (n_shots + n_queries), dtype=bool)
    mask[support_indices] = False
    query_indices = torch.arange(n_ways * (n_shots + n_queries))[mask]

    return support_indices, query_indices


def get_model_optimizer(args):
    model = eval(args.model)(args)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    else:
        raise Exception("Unknown optimizer")

    if args.lr_scheduler:
        if args.lr_scheduler == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(args.step_size),
                gamma=args.gamma
            )
        elif args.lr_scheduler == 'multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(_) for _ in args.step_size.split(',')],
                gamma=args.gamma,
            )
        elif args.lr_scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.max_epoch,
                eta_min=0  # a tuning parameter
            )
        else:
            raise ValueError('Unknown Scheduler')
    else:
        lr_scheduler = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['train_epoch']
        args.start_step = checkpoint['train_step']

    return model, optimizer, lr_scheduler


if __name__ == '__main__':
    s, q = partition_task(5, 1, 5)
    print(s,q)
