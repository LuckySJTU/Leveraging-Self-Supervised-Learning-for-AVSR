import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
import numpy as np
import torch

from .utils import prepare_pretrain_input, prepare_main_input
from .vision_transform import ToTensor, Normalize, RandomCrop, CenterCrop, RandomHorizontalFlip


class Voxceleb2(Dataset):
    """
    A custom dataset class for the Voxceleb2 dataset.
    """

    def __init__(self, modal, dataset, datadir, h5file, charToIx, stepSize, lrs2Aug, noiseParams):
        super(Voxceleb2, self).__init__()
        self.dataset = dataset
        self.stepSize = stepSize
        self.h5file = h5file
        self.transform = transforms.Compose([
            ToTensor(),
            RandomCrop(112),
            RandomHorizontalFlip(0.5),
        ])

    def open_h5(self):
        self.h5 = h5py.File(self.h5file, "r")

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        if self.dataset == "train":
            # index goes from 0 to stepSize-1
            # dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            # fetch the sample at position 'index' in this randomly selected partition
            base = self.stepSize * np.arange(int(1091721 / self.stepSize) + 1)
            ixs = base + index
            ixs = ixs[ixs < 1091721]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)
        if self.dataset == "val":
            index += 1091721
        elif self.dataset == "test":
            index += 1091721

        vidInp = cv.imdecode(self.h5["png"][index], cv.IMREAD_COLOR)
        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
        vidInp = torch.tensor(vidInp).unsqueeze(1)
        vidInp = self.transform(vidInp)
        return vidInp, None

    def __len__(self):
        # each iteration covers only a random subset of all the training samples whose size is given by the step size
        # this is done only for the pretrain set, while the whole val/test set is considered
        if self.dataset == "train":
            return self.stepSize
        else:
            return 36237
