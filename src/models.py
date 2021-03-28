import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import pandas as pd
from PIL import Image
import utils
import os, io, sys, h5py, shutil
import numpy as np
from time import gmtime, strftime


class ACP(Dataset):
    def __init__(self, paths, labels, h5file=None, transform=None):
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
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            utils.prepare_csv(self.h5trainfile, self.csv_train)
            if self.h5testfile:
                dirname = os.path.dirname(self.csv_test)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                utils.prepare_csv(self.h5testfile, self.csv_test)

    def setup(self, stage=None):
        phases = ["train", "val", "test"]

        if not self.datasplitdir:
            splits = utils.train_val_test_split(
                *utils.read_from_csv(self.csv_train),
                val_size=0.15 if not self.h5testfile else 0.2,
                test_size=0.15 if not self.h5testfile else None,
                rs_train=69, rs_test=69
            )
            if self.h5testfile:
                splits.update({"test": utils.read_from_csv(self.csv_test)})
            shutil.rmtree(os.path.dirname(self.csv_train))
            data_splits = "../data/data_splits"
            currenttime = strftime("%Y-%m-%dT%Hh%Mm%Ss", gmtime())
            split_save_dir = os.path.join(data_splits, currenttime)
            if not os.path.exists(split_save_dir):
                os.makedirs(split_save_dir)

            for phase in phases:
                np.save(f"{split_save_dir}/{phase}", splits[phase])

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


class LinearModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.in_features = input_dim
        self.flatten = nn.Flatten()
        self.out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.out(x)


class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        model_class="LogisticRegression",
        input_dim=(3, 64, 64),
        use_vae=False,
        learning_rate=0.001,
        stepsize=5,
        gamma=0.5,
        hinge_deg=1,
        weight_decay=1e-3,
    ):
        super().__init__()
        self.num_classes = num_classes
        c, w, h = input_dim
        self.dims = c * w * h
        self.save_hyperparameters("learning_rate", "stepsize", "gamma", "weight_decay", "hinge_deg")
        self.example_input_array = torch.rand(1, 3, 64, 64, device=self.device)
        latent_dim = self.dims if not use_vae else 512
        if model_class == "LogisticRegression":
            self.criterion = nn.CrossEntropyLoss()
        elif model_class == "SVM":
            self.criterion = nn.MultiMarginLoss(p=self.hparams.hinge_deg)
        self.model_class = model_class
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.backbone = []
        if use_vae:
            from pl_bolts.models.autoencoders import VAE

            vae_ckpt = (
                "C:\\Users\\towab\\.cache\\torch\\hub\\checkpoints\\epoch=89.ckpt"
            )
            vae = VAE(input_height=64).load_from_checkpoint(vae_ckpt)
            vae.freeze()
            self.backbone.append(vae)
        self.backbone.append(LinearModel(latent_dim, self.num_classes))
        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.train_acc(preds, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.hparams.stepsize, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


class MLPTorch(nn.Module):
    def __init__(self, num_classes: int = 3, input_dim: tuple = (3, 64, 64)):
        super().__init__()
        channels, width, height = input_dim
        input_dim = channels * width * height
        self.flatten = nn.Flatten()
        self.block1 = self.linear_block(input_dim, input_dim // 256, dropout_p=0.3)
        self.block2 = self.linear_block(input_dim // 256, input_dim // 512, dropout_p=0.2)
        self.block3 = self.linear_block(input_dim // 512, input_dim // 1024, dropout_p=0.1)
        self.out = nn.Linear(input_dim // 1024, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out(x)

    def linear_block(self, d_in, d_out, dropout_p=0.1, batchnorm=True):
        modules = []
        modules.append(nn.Linear(d_in, d_out))
        if batchnorm:
            modules.append(nn.BatchNorm1d(d_out))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout_p))
        return nn.Sequential(*modules)


class MLP(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        input_dim=(3, 64, 64),
        learning_rate=0.001,
        stepsize=5,
        gamma=0.5,
        weight_decay=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters("learning_rate", "stepsize", "gamma", "weight_decay")
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.example_input_array = torch.rand(1, 3, 64, 64, device=self.device)
        self.backbone = MLPTorch(num_classes=num_classes, input_dim=input_dim)

    def forward(self, x):
        return F.softmax(self.backbone(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.train_acc(preds, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.hparams.stepsize, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


class ResnetTorch(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_extract=True,
        pretrained=True,
        resnet_base=models.resnet50(pretrained=True),
    ):
        super().__init__()
        self.feature_extractor = resnet_base
        self.set_parameter_requires_grad(freeze=feature_extract)
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = LinearModel(num_features, num_classes)

    def forward(self, x):
        return self.feature_extractor(x)

    def set_parameter_requires_grad(self, freeze: bool, layers: list = None):
        for name, param in self.named_parameters():
            if not layers:
                param.requires_grad = not freeze
            elif any(x in name for x in layers):
                param.requires_grad = not freeze


class Resnet(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        input_dim=(3, 64, 64),
        feature_extract=True,
        learning_rate=0.001,
        stepsize=5,
        gamma=0.5,
        weight_decay=5e-4,
        resnet_base=models.resnet50(pretrained=True),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dims = input_dim
        self.example_input_array = torch.rand(1, 3, 64, 64, device=self.device)
        self.feature_extract = feature_extract
        self.save_hyperparameters("learning_rate", "stepsize", "gamma", "weight_decay")
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.resnet = ResnetTorch(self.num_classes, resnet_base=resnet_base)

    def forward(self, x):
        return F.softmax(self.resnet(x), dim=1)

    def configure_optimizers(self):
        # opt_ft_ext = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        #     momentum=0.9
        # )
        opt_finetuning = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # base_scheduler = torch.optim.lr_scheduler.StepLR(
        #     opt_finetuning, self.hparams.stepsize, gamma=self.hparams.gamma
        # )
        # sched_ft_ext = {
        #     'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ft_ext, T_0=self.hparams.stepsize),
        #     'interval': 'epoch'
        # }
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_finetuning, mode='max', patience=self.hparams.stepsize
        )
        sched_finetuning = {
            'scheduler': lr_plateau,
            'reduce_on_plateau': True,
            'monitor': 'val_acc'
        }
        # if self.feature_extract:
        #     return [opt_ft_ext], [sched_ft_ext]
        # else:
        # return [opt_finetuning], [sched_finetuning]
        return {'optimizer': opt_finetuning, 'lr_scheduler': sched_finetuning}

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.resnet(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.train_acc(preds, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.resnet(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.resnet(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def set_parameter_requires_grad(self, freeze: bool, layers: list = None):
        self.feature_extract = freeze
        self.resnet.set_parameter_requires_grad(freeze=freeze, layers=layers)
