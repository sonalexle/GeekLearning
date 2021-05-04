import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
import utils
import os, io, sys, h5py, shutil
import numpy as np
from time import gmtime, strftime


class ACP(Dataset):
    """
    Subclass of torch.utils.data.Dataset, handles images stored in hdf5 files.
    The images in the hdf5 file should be arranged in this way:
    class1/a.png
    class1/[...]/c.png

    class2/a.png
    class2/[...]/d.png
    """
    def __init__(self, paths, labels, h5file=None, transform=None):
        """
        Parameters
        ----------
        paths: array-like
            paths to images (within the hdf5 file)
        labels: array-like
            labels corresponding to the paths
        h5file: h5py.File, optional
            The hdf5 file. Must be open.
        """
        self.X = paths
        self.y = labels
        self.h5file = h5file
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if not self.h5file:
            image = Image.open(x)
        else:
            imgbyte = np.array(self.h5file[str(y)][x])
            image = Image.open(io.BytesIO(imgbyte))
        label = torch.tensor(y, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


def loader_from_h5(h5file, h5path, batch_size=64, num_workers=0, transform=None):
    """Creates a DataLoader from a hdf5 file."""
    paths_and_labels = utils.prepare_csv(h5file, h5path, write=False)
    paths, labels = utils.read_from_csv(df=paths_and_labels)
    if not transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return DataLoader(
        ACP(paths, labels, h5file, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


class ACPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5trainfile,
        h5testfile=None,
        datasplitdir=None,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_train=False,
        augment="all",
        eval=False,
    ):
        """
        Args:
            h5testfile: Optional hdf5 test file. If None (default), the trainset 
                will consist of 70% train, 15% validation, and 15% test. 
                If supplied, the trainset will consist of 80% train and 20% validation.  
            datasplitdir: Optional path to a data split directory 
                created in previous train sessions by objects of class `ACPDAtaModule`. 
                If supplied, the object will use data splits specified by the files 
                in this directory. Inside the directory there must be three files: 
                train.npy, val.npy, and test.npy.
        """
        if datasplitdir:
            assert os.path.exists(datasplitdir), "Wrong data directory"
        self.datasplitdir = datasplitdir
        self.h5trainfile = h5trainfile
        self.h5testfile = h5testfile
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.csv_train = "./lookup_csvs/imageclasslookup_train.csv"
        self.csv_test = "./lookup_csvs/imageclasslookup_test.csv"
        random_transforms = [
            transforms.RandomGrayscale(),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(),
        ]
        if augment == "all":
            rand_trans = transforms.RandomOrder(random_transforms)
        else:
            rand_trans = transforms.RandomChoice(random_transforms)
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        train_transforms = base_transforms
        if not eval:
            train_transforms = [rand_trans] + base_transforms
            train_transforms.append(transforms.RandomErasing(p=0.1))
        self.transforms = {
            "train": transforms.Compose(train_transforms),
            "val": transforms.Compose(base_transforms),
        }
        self.dims = (3, 64, 64)
        self.num_classes = 3

    def prepare_data(self):
        if not self.datasplitdir:
            dirname = os.path.dirname(self.csv_train)
            if not os.path.exists(dirname): os.makedirs(dirname)
            utils.prepare_csv(self.h5trainfile, self.csv_train)
            if self.h5testfile:
                dirname = os.path.dirname(self.csv_test)
                if not os.path.exists(dirname): os.makedirs(dirname)
                utils.prepare_csv(self.h5testfile, self.csv_test)

    def setup(self, stage=None, rs_train=69, rs_test=69):
        phases = ["train", "val", "test"]

        # performs the splits if not supplied
        if not self.datasplitdir:
            splits = utils.train_val_test_split(
                *utils.read_from_csv(self.csv_train),
                val_size=0.15 if not self.h5testfile else 0.2,
                test_size=0.15 if not self.h5testfile else None,
                rs_train=rs_train, rs_test=rs_test
            )
            # if h5testfile is supplied, add it to the splits
            # (currently consisting of train and val)
            if self.h5testfile:
                splits.update({"test": utils.read_from_csv(self.csv_test)})

            # delete the generated csv files
            shutil.rmtree(os.path.dirname(self.csv_train))

            # saves the splits for later use
            data_splits = "../data/data_splits"
            currenttime = strftime("%Y-%m-%dT%Hh%Mm%Ss", gmtime())
            split_save_dir = os.path.join(data_splits, currenttime)
            if not os.path.exists(split_save_dir): os.makedirs(split_save_dir)
            for phase in phases:
                np.save(f"{split_save_dir}/{phase}", splits[phase])

        # if splits are known, load from the supplied path
        else:
            splits = {
                phase: np.load(f"{self.datasplitdir}/{phase}.npy", allow_pickle=True)
                for phase in phases
            }

        if stage == "fit" or stage is None:
            self.acp_train = ACP(
                *splits["train"], self.h5trainfile, transform=self.transforms["train"]
            )
            self.acp_val = ACP(
                *splits["val"], self.h5trainfile, transform=self.transforms["val"]
            )

        if stage == "test" or stage is None:
            file = self.h5trainfile
            if self.h5testfile:
                file = self.h5testfile
            self.acp_test = ACP(*splits["test"], file, transform=self.transforms["val"])

    def train_dataloader(self):
        return DataLoader(
            self.acp_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.acp_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.acp_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

