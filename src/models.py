import torchvision, torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


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
        learning_rate=0.001,
        stepsize=5,
        gamma=0.5,
        hinge_deg=1,
        weight_decay=1e-3,
        use_vae=False
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
        self.softmax = nn.Softmax(dim=1)
        self.main = []
        # if use_vae:
        #     from pl_bolts.models.autoencoders import VAE

        #     vae_ckpt = (
        #         "C:\\Users\\towab\\.cache\\torch\\hub\\checkpoints\\epoch=89.ckpt"
        #     )
        #     vae = VAE(input_height=64).load_from_checkpoint(vae_ckpt)
        #     vae.freeze()
        #     self.main.append(vae)
        self.main.append(LinearModel(latent_dim, self.num_classes))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.softmax(self.main(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.train_acc(preds, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
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
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.main = MLPTorch(num_classes=num_classes, input_dim=input_dim)

    def forward(self, x):
        return self.softmax(self.main(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.train_acc(preds, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
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
        resnet_base=torchvision.models.resnet50(pretrained=True)
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
        resnet_base=torchvision.models.resnet50(pretrained=True),
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
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.main = ResnetTorch(self.num_classes, resnet_base=resnet_base)

    def forward(self, x):
        return self.softmax(self.main(x))

    def configure_optimizers(self):
        opt_finetuning = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_finetuning, mode='max', patience=self.hparams.stepsize,
            factor=self.hparams.gamma
        )
        sched_finetuning = {
            'scheduler': lr_plateau,
            'reduce_on_plateau': True,
            'monitor': 'val_acc'
        }
        return {'optimizer': opt_finetuning, 'lr_scheduler': sched_finetuning}

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.train_acc(preds, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.main(x)
        loss = self.criterion(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def set_parameter_requires_grad(self, freeze: bool, layers: list = None):
        self.feature_extract = freeze
        self.main.set_parameter_requires_grad(freeze=freeze, layers=layers)