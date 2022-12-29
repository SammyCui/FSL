import os
from typing import List, Iterator, Tuple

import torch
import numpy as np
import learn2learn as l2l
from torchvision import transforms
from data.datasets.cub import CUBirds200
from data.datasets.mini_imagenet import MiniImageNet
from data.datasets.omniglot import OmniglotFull
from learn2learn.data import transforms as l2l_transforms, MetaDataset, TaskDataset
from torch.utils.data import DataLoader

IMAGENET_NORMALIZE = {"mean": [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                      "std": [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]}


def _tasks_sampler(dataset_name: str,
                   root: str,
                   n_ways_train: int,
                   n_shots_train: int,
                   n_queries_train: int,
                   n_ways_test: int,
                   n_shots_test: int,
                   n_queries_test: int,
                   episodes_per_epoch: int = 100,
                   num_val_episodes: int = 600,
                   num_test_episodes: int = 10000,
                   download: bool = True) -> Tuple[Iterator, Iterator, Iterator]:
    if dataset_name == 'omniglot':
        root_omniglot = root
        classes = list(range(1623)) # 1623 classes in omniglot
        np.random.seed(40)
        np.random.shuffle(classes)

        def transform(x):
            return 1.0 - x

        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(28), #, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Lambda(transform),
        ])

        dataset = OmniglotFull(root_omniglot, download=download,
                               transform=image_transforms, target_transform=None)
        dataset = MetaDataset(dataset)
        train_trans = l2l_transforms.FusedNWaysKShots(dataset,
                                                      n=n_ways_train,
                                                      k=n_shots_train + n_queries_train,
                                                      filter_labels=classes[:1100])
        eval_trans = l2l_transforms.FusedNWaysKShots(dataset,
                                                     n=n_ways_test,
                                                     k=n_shots_test + n_queries_test,
                                                     filter_labels=classes[1100:1200])
        test_trans = l2l_transforms.FusedNWaysKShots(dataset,
                                                     n=n_ways_test,
                                                     k=n_shots_test + n_queries_test,
                                                     filter_labels=classes[1200:])
        task_transforms = [
            l2l_transforms.LoadData(dataset),
            l2l_transforms.RemapLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0]),

        ]
        dataset = MetaDataset(dataset)
        train_tasks = TaskDataset(dataset, task_transforms=[train_trans] + task_transforms, num_tasks=episodes_per_epoch)
        eval_tasks = TaskDataset(dataset, task_transforms=[eval_trans] + task_transforms, num_tasks=num_val_episodes)
        test_tasks = TaskDataset(dataset, task_transforms=[test_trans] + task_transforms, num_tasks=num_test_episodes)

    elif dataset_name == 'mini_imagenet':
        root_mini_imagenet = os.path.join(root, 'mini-imagenet')
        if not os.path.exists(root_mini_imagenet):
            os.makedirs(root_mini_imagenet)
        train_image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(**IMAGENET_NORMALIZE)
        ])

        eval_image_transforms = transforms.Compose([
            transforms.Normalize(**IMAGENET_NORMALIZE)
        ])

        train_dataset = MiniImageNet(root_mini_imagenet, 'train', transform=train_image_transforms,
                                     target_transform=None, download=download)
        eval_dataset = MiniImageNet(root_mini_imagenet, 'validation', transform=eval_image_transforms,
                                    target_transform=None, download=download)
        test_dataset = MiniImageNet(root_mini_imagenet, 'test', transform=eval_image_transforms, target_transform=None,
                                    download=download)

        train_dataset = MetaDataset(train_dataset)
        eval_dataset = MetaDataset(eval_dataset)
        test_dataset = MetaDataset(test_dataset)

        train_task_transforms = [l2l_transforms.FusedNWaysKShots(train_dataset,
                                                                 n=n_ways_train,
                                                                 k=n_shots_train + n_queries_train),
                                 l2l_transforms.LoadData(train_dataset),
                                 l2l_transforms.RemapLabels(train_dataset),
                                 ]
        eval_task_transforms = [l2l_transforms.FusedNWaysKShots(eval_dataset,
                                                                n=n_ways_test,
                                                                k=n_shots_test + n_queries_test),
                                l2l_transforms.LoadData(eval_dataset),
                                l2l_transforms.RemapLabels(eval_dataset),
                                ]
        test_task_transforms = [l2l_transforms.FusedNWaysKShots(test_dataset,
                                                                n=n_ways_test,
                                                                k=n_shots_test + n_queries_test),
                                l2l_transforms.LoadData(test_dataset),
                                l2l_transforms.RemapLabels(test_dataset),
                                ]

        train_tasks = TaskDataset(train_dataset, task_transforms=train_task_transforms, num_tasks=episodes_per_epoch)
        eval_tasks = TaskDataset(eval_dataset, task_transforms=eval_task_transforms, num_tasks=num_val_episodes)
        test_tasks = TaskDataset(test_dataset, task_transforms=test_task_transforms, num_tasks=num_test_episodes)

    elif dataset_name == 'cub':
        root_cub = os.path.join(root, 'cub')

        train_image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_NORMALIZE)
        ])

        eval_image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**IMAGENET_NORMALIZE)
        ])
        train_dataset = CUBirds200(root_cub, 'train', transform=train_image_transforms,
                                   target_transform=None, download=download)
        eval_dataset = CUBirds200(root_cub, 'validation', transform=eval_image_transforms,
                                  target_transform=None, download=download)
        test_dataset = CUBirds200(root_cub, 'test', transform=eval_image_transforms,
                                  target_transform=None, download=download)

        train_dataset = MetaDataset(train_dataset)
        eval_dataset = MetaDataset(eval_dataset)
        test_dataset = MetaDataset(test_dataset)

        train_task_transforms = [l2l_transforms.FusedNWaysKShots(train_dataset,
                                                                 n=n_ways_train,
                                                                 k=n_shots_train + n_queries_train),
                                 l2l_transforms.LoadData(train_dataset),
                                 l2l_transforms.RemapLabels(train_dataset),
                                 ]
        eval_task_transforms = [l2l_transforms.FusedNWaysKShots(eval_dataset,
                                                                n=n_ways_test,
                                                                k=n_shots_test + n_queries_test),
                                l2l_transforms.LoadData(eval_dataset),
                                l2l_transforms.RemapLabels(eval_dataset),
                                ]
        test_task_transforms = [l2l_transforms.FusedNWaysKShots(test_dataset,
                                                                n=n_ways_test,
                                                                k=n_shots_test + n_queries_test),
                                l2l_transforms.LoadData(test_dataset),
                                l2l_transforms.RemapLabels(test_dataset),
                                ]

        train_tasks = TaskDataset(train_dataset, task_transforms=train_task_transforms, num_tasks=episodes_per_epoch)
        eval_tasks = TaskDataset(eval_dataset, task_transforms=eval_task_transforms, num_tasks=num_val_episodes)
        test_tasks = TaskDataset(test_dataset, task_transforms=test_task_transforms, num_tasks=num_test_episodes)

    else:
        raise Exception("Dataset not implemented")

    return train_tasks, eval_tasks, test_tasks


def task_loader(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tasks, val_tasks, test_tasks = _tasks_sampler(args.dataset_name,
                                                        root=args.root,
                                                        n_ways_train=args.n_ways_train,
                                                        n_shots_train=args.n_shots_train,
                                                        n_queries_train=args.n_queries_train,
                                                        n_ways_test=args.n_ways_test,
                                                        n_shots_test=args.n_shots_test,
                                                        n_queries_test=args.n_queries_test,
                                                        episodes_per_epoch=args.episodes_per_epoch,
                                                        num_val_episodes=args.num_val_episodes,
                                                        num_test_episodes=args.num_test_episodes,
                                                        download=args.download)
    train_dataloader = DataLoader(train_tasks, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_tasks, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_dataloader = DataLoader(test_tasks, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

