from collections import OrderedDict

import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.backbone == 'convnet':
            from models.backbones.convnet import ConvNet
            if args.dataset_name == 'omniglot':
                self.encoder = ConvNet(in_channels=1, global_pool=None)
            else:
                self.encoder = ConvNet()
        elif args.backbone == 'resnet12':
            from models.backbones.resnet import Resnet12
            if args.dataset_name == 'omniglot':
                self.encoder = Resnet12(in_channels=1)
            else:
                self.encoder = Resnet12()
        elif args.backbone == 'wrn28':
            from models.backbones.wrn import WRN
            assert not args.dataset_name == 'omniglot', 'WRN does not support Omniglot'

            self.encoder = WRN(28, 10, 0.5)  # 0.5 used in FEAT
        else:
            raise ValueError("Backbone not implemented")

        if args.init_backbone:
            encoder_pretrained_params = OrderedDict(
                {'encoder.' + k: v for k, v in torch.load(args.init_backbone).items()})

            self.encoder.load_state_dict(encoder_pretrained_params)

        # to be initialized in the trainer
        self.support_idx_train, self.query_idx_train, self.support_idx_test, self.query_idx_test = (None,) * 4

    def forward(self, x):
        x = x.squeeze(0)
        embeddings = self.encoder(x)

        logits = self._forward(embeddings)
        return logits

    def _forward(self, x):
        raise NotImplementedError('Need to be implemented')
