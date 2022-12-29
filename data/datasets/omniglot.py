import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot
import torchvision


class OmniglotFull(Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/full_omniglot.py)
    **Description**
    This class provides an interface to the Omniglot dataset.
    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(self.root, background=True, download=download)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(self.root,
                                   background=False,
                                   download=download,
                                   target_transform=torchvision.transforms.Lambda(self.label_transform))

        self.dataset = ConcatDataset((omni_background, omni_evaluation))
        self._bookkeeping_path = os.path.join(self.root, 'omniglot-bookkeeping.pkl')

    @staticmethod
    def label_transform(x):
        return x + 964

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

if __name__ == '__main__':
    omniglot = OmniglotFull(root = "../data", download=True)
    print(len(omniglot))