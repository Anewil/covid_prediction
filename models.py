from torchvision import transforms
from torchvision.datasets import ImageFolder
from squeezenet import SqueezeNet

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomAffine((-20, 20), translate=None, scale=[0.7, 1.3], shear=None, resample=False, fillcolor=0),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class PneumoniaModel(SqueezeNet):

    def prepare_data(self):
        self.train_set = ImageFolder('chest_xray/train', transform=transform_train)
        self.test_set = ImageFolder('chest_xray/test', transform=transform)
        self.val_set = ImageFolder('chest_xray/val', transform=transform)


class CoronahackModel(SqueezeNet):

    def prepare_data(self):
        self.train_set = ImageFolder('virusorbacteria/train', transform=transform_train)
        self.test_set = ImageFolder('virusorbacteria/test', transform=transform)
        self.val_set = ImageFolder('virusorbacteria/val', transform=transform)


class CovidModel(SqueezeNet):

    def prepare_data(self):
        self.train_set = ImageFolder('covid/train', transform=transform_train)
        self.test_set = ImageFolder('covid/test', transform=transform)
        self.val_set = ImageFolder('covid/val', transform=transform)
