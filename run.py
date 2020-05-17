import torch
from pytorch_lightning import Trainer
from models import PneumoniaModel, CoronahackModel, CovidModel
from torch.utils.tensorboard import SummaryWriter


class HParams:
    def __init__(self, params):
        self.__dict__.update(**params)


if __name__ == '__main__':
    writer = SummaryWriter()
    params = {
        'max_epochs': 100,
        'fast_dev_run': True,
        'batch_size': 50,
        'learning_rate': 0.0003,
    }
    hparams = HParams(params)
    # pneumonia_model = PneumoniaModel(hparams)
    # trainer = Trainer(max_epochs=hparams.max_epochs,
    #                   fast_dev_run=hparams.fast_dev_run,
    #                   gpus=1
    #                   )
    # trainer.fit(pneumonia_model)
    # trainer.test()
    # pneumonia_model.plot_confusion_matrix()
    coronahack_model = CoronahackModel(hparams).load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=92.ckpt')
    trainer = Trainer(max_epochs=hparams.max_epochs,
                      fast_dev_run=hparams.fast_dev_run,
                      gpus=1
                      )
    trainer.fit(coronahack_model)
    trainer.test()
    dataloader = coronahack_model.train_dataloader()
    # coronahack_model.plot_confusion_matrix()
    # covid_model = CovidModel(hparams).load_from_checkpoint('lightning_logs/version_1/checkpoints/epoch=37.ckpt')
    # trainer = Trainer(max_epochs=hparams.max_epochs,
    #                   fast_dev_run=hparams.fast_dev_run,
    #                   gpus=1
    #                   )
    # trainer.fit(covid_model)
    # trainer.test()
    # covid_model.plot_confusion_matrix()
