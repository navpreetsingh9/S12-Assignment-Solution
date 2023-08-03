from pytorch_lightning import LightningModule, seed_everything
from torchmetrics import Accuracy
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from cifar10.model import CustomResNet
from cifar10.dataloader import Cifar10SearchDataset
from cifar10.transforms import CustomResnetTransforms

from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder

def get_sgd_optimizer(model, lr, momentum=0.9, weight_decay=5e-4):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_adam_optimizer(model, lr, weight_decay=1e-4):
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_lr_finder(model, optimizer, criterion, train_loader, end_lr, device="cuda"):
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=200, step_mode="exp")
    _, suggested_lr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return suggested_lr


def get_onecyclelr_scheduler(optimizer, max_lr, train_loader, epochs):
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=5/epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )

class LitCifar10(LightningModule):
    def __init__(self, data_dir="./data", learning_rate=0.03, weight_decay=1e-4, end_lr=10, batch_size=256):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]

        self.train_transforms = CustomResnetTransforms.train_transforms(means, stds)
        self.test_transforms = CustomResnetTransforms.test_transforms(means, stds)

        # Define PyTorch model
        self.model = CustomResNet()

        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)

        preds = torch.argmax(y_pred, dim=1)
        self.accuracy(preds, target)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_loss", loss, prog_bar=True, logger=False)
        self.log("train_acc", self.accuracy, prog_bar=True, logger=False)
        self.log("lr", cur_lr, prog_bar=True, logger=False)
        self.logger.experiment.add_scalars('loss', {'train': loss}, self.current_epoch)
        self.logger.experiment.add_scalars('acc', {'train': self.accuracy(preds, target)}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        preds = torch.argmax(output, dim=1)
        self.accuracy(preds, target)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, logger=False)
        self.log("val_acc", self.accuracy, prog_bar=True, logger=False)
        self.logger.experiment.add_scalars('loss', {'valid': loss}, self.current_epoch)
        self.logger.experiment.add_scalars('acc', {'valid': self.accuracy(preds, target)}, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        preds = torch.argmax(output, dim=1)
        self.accuracy(preds, target)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = get_adam_optimizer(self.model, self.lr, weight_decay=self.weight_decay)
        max_lr = get_lr_finder(self.model, optimizer, self.criterion, self.train_dataloader(), self.end_lr)
        lr_scheduler = get_onecyclelr_scheduler(optimizer, max_lr, train_loader=self.train_dataloader(), epochs=self.trainer.max_epochs)
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    ####################
    # DATA RELATED HOOKS
    ####################


    def setup(self, stage=None):

        seed_everything(42, workers=True)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_train = Cifar10SearchDataset(self.data_dir, train=True, download=True, transform=self.train_transforms)
            self.cifar_val = Cifar10SearchDataset(self.data_dir, train=False, download=True, transform=self.test_transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = Cifar10SearchDataset(self.data_dir, train=False, download=True, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=os.cpu_count())