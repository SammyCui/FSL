import argparse
import os.path

import torch


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', type=str, default='ProtoNet', choices=['ProtoNet'])
    parser.add_argument('--backbone_class', type=str, default='convnet',
                        choices=['convnet', 'resnet12', 'wrn28'])
    parser.add_argument('--dataset_name', type=str, default='mini_imagenet',
                        choices=['mini_imagenet', 'omniglot', 'cub'])
    parser.add_argument('--root', type=str, default='./data/data')

    parser.add_argument('--n_ways_train', type=int, default=5)
    parser.add_argument('--n_shots_train', type=int, default=1)
    parser.add_argument('--n_queries_train', type=int, default=15)
    parser.add_argument('--n_ways_test', type=int, default=5)
    parser.add_argument('--n_shots_test', type=int, default=1)
    parser.add_argument('--n_queries_test', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=1)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_val_episodes', type=int, default=600)
    parser.add_argument('--num_test_episodes', type=int, default=10000)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['multistep', 'step', 'cosine'], nargs='?', const=None)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.2)  # for lr_scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # we find this weight decay value works the best

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='gpu')
    # parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--download', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--save', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--resume', type=str, default=None, nargs='?', const=None)
    parser.add_argument('--init_backbone', type=str, default=None, nargs='?', const=None)

    args = parser.parse_args()
    return args


def process_args(args):
    assert not (args.init_backbone and args.resume), "Can't have both resume and init_backbone"
    args.download = eval(args.download)
    args.save = eval(args.save)
    if args.device == 'gpu':
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            raise Exception("No GPU available")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.root):
        os.makedirs(args.root)


class DebugArgs:
    def __init__(self,
                 model: str,
                 backbone: str,
                 dataset_name: str,
                 n_ways_train: int,
                 n_shots_train: int,
                 n_queries_train: int,
                 n_ways_test: int,
                 n_shots_test: int,
                 n_queries_test: int,
                 root: str = './data/data',
                 episodes_per_epoch: int = 100,
                 temperature: int = 1,
                 start_epoch: int = 0,
                 max_epoch: int = 200,
                 num_val_episodes: int = 600,
                 num_test_episodes: int = 10000,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 lr_scheduler: str = 'step',
                 step_size: int = 20,
                 gamma: float = 0.2,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 val_interval: int = 1,
                 num_workers: int = 4,
                 download: bool = False,
                 device: str = 'gpu',
                 result_dir: str = './checkpoints',
                 save: bool = False,
                 resume: bool = False,
                 init_backbone: bool = False):

        self.model = model
        self.root = root
        self.backbone = backbone
        self.dataset_name = dataset_name
        self.n_ways_train = n_ways_train
        self.n_shots_train = n_shots_train
        self.n_queries_train = n_queries_train
        self.n_ways_test = n_ways_test
        self.n_shots_test = n_shots_test
        self.n_queries_test = n_queries_test
        self.download = download
        self.num_workers = num_workers
        self.episodes_per_epoch = episodes_per_epoch
        self.temperature = temperature
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes
        self.lr = lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.val_interval = val_interval
        self.device = device
        self.result_dir = result_dir
        self.save = save
        self.resume = resume
        self.init_backbone = init_backbone
