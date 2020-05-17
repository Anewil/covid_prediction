import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from torchvision.models import squeezenet1_1
import matplotlib.pyplot as plt
import numpy as np


class SqueezeNet(LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        super(SqueezeNet, self).__init__()
        self.hparams = hparams
        self.y_trues = []
        self.predictions = []
        self.model = squeezenet1_1(pretrained=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_trues, self.predictions)
        plt.figure()
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=self.test_set.class_to_idx.keys(),
               yticklabels=self.test_set.class_to_idx.keys(),
               xlabel='True label',
               ylabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i][j], ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        plt.show()

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        loss = loss.unsqueeze(0)

        tqdm_dict = {'train_loss': loss}
        return {'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._test_step(batch, batch_idx, 'test')

    def _test_step(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        self.y_trues.extend(y.data.unsqueeze(0).cpu().numpy().flatten())
        self.predictions.extend(labels_hat.unsqueeze(0).cpu().numpy().flatten())
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)

        if self.on_gpu:
            acc = acc.cuda(loss.device.index)

        loss = loss.unsqueeze(0)
        acc = acc.unsqueeze(0)

        return {f'{prefix}_loss': loss, f'{prefix}_acc': acc}

    def _eval_step(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)

        if self.on_gpu:
            acc = acc.cuda(loss.device.index)

        loss = loss.unsqueeze(0)
        acc = acc.unsqueeze(0)

        return {f'{prefix}_loss': loss, f'{prefix}_acc': acc}

    def validation_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'test')

    def _eval_epoch_end(self, outputs, prefix):
        loss_mean = 0
        acc_mean = 0
        for output in outputs:
            loss = output[f'{prefix}_loss']

            loss = torch.mean(loss)
            loss_mean += loss

            # reduce manually when using dp
            acc = output[f'{prefix}_acc']
            acc = torch.mean(acc)

            acc_mean += acc

        loss_mean /= len(outputs)
        acc_mean /= len(outputs)
        tqdm_dict = {f'{prefix}_loss': loss_mean, f'{prefix}_acc': acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, f'{prefix}_loss': loss_mean}
        return result

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def prepare_data(self):
        pass

    def make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight

    def train_dataloader(self):
        weights = self.make_weights_for_balanced_classes(self.train_set.imgs, len(self.train_set.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=4,
                          sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=4, shuffle=False)
