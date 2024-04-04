import torch
from torch import nn
import torchvision as tv
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

import pltraining
from pltraining import parse_yaml, construct_params

from datamodule import WCDataModule
from model import EffDetWordChar, EASTWCLoss
import utils

# Module

def make_grid(scores, images, scale):
    
    drawn_images = []
    for score, image in zip(scores, images):
        polys = utils.decode(score, scale=scale, ths=0.8, iou_nms=0.3)
        ia_polys = utils.polys_to_imgaug(polys, shape=image.shape[-1:0:-1])
        im = image.permute(1, 2, 0).numpy() * .5 + .5
        drawn_polys = ia_polys.draw_on_image((im*255).astype(np.uint8))
        
        drawn_images.append(tv.transforms.ToTensor()(drawn_polys))
        
    x = torch.stack(drawn_images)
    grid = tv.utils.make_grid(x, nrow=2)
    return grid

class EASTWCTuner(pltraining.EASTUner):
    
    def __init__(self, hparams=None, **kwargs):

        self.model_kwargs = kwargs
        self.hparams = self._construct_hparams(hparams)

        super(pltraining.EASTUner, self).__init__()

        self.model = EffDetWordChar(advprop=self.hparams.advprop,
                                    compound_coef=self.hparams.coef,
                                    expand_bifpn=self.hparams.expand_bifpn,
                                    factor2=self.hparams.factor2,
                                    repeat_bifpn=self.hparams.repeat_bifpn,
                                    bifpn_channels=self.hparams.bifpn_channels,)
        
        self.loss_fct = self.get_loss_fct()
        
    def get_loss_fct(self,):
        return EASTWCLoss(**self.hparams.loss_hparams)
    
    def _handle_batch(self, batch):
        image, wgt, cgt = batch
        word_scores, char_scores = self(image)

        word_loss, word_losses = self.loss_fct(wgt.float(), word_scores)
        char_loss, char_losses = self.loss_fct(cgt.float(), char_scores)
        
        loss = word_loss + char_loss
        
        
        word_losses = {
            f'word_{k}':v for k, v in char_losses.items()
        }
        char_losses = {
            f'char_{k}':v for k, v in char_losses.items()
        }
        
        losses = dict(word_loss=word_loss, char_loss=char_loss, **word_losses, **char_losses)
        
        return (loss, [word_scores, char_scores], losses)
    
    def validation_step(self, batch, batch_idx):
        loss, scores, _ = self._handle_eval_batch(batch)
        if batch_idx == 0:
            try:
                
                images = batch[0]
                word_scores, char_scores = scores
                
                word_grid = make_grid(word_scores, images, scale=self.hparams.scale)
                char_grid = make_grid(char_scores, images, scale=self.hparams.scale)
                
                self.logger.experiment.log_image(
                    'word_image', tv.transforms.ToPILImage()(word_grid))
                self.logger.experiment.log_image(
                    'char_image', tv.transforms.ToPILImage()(char_grid))
            except:
                pass
        return {'val_loss': loss}


if __name__ == '__main__':

    # HPARAMS
    hparams = parse_yaml('hparams.yaml')
    params = construct_params(hparams)

    # DATA
    data_module = WCDataModule(**hparams['datamodule'])

    # MODEL
    model_tuner = EASTWCTuner(**hparams['model'])


    # NEPTUNE
    neptune_logger = NeptuneLogger(
        project_name='israelcamp/TextDetection',
        experiment_name='effdet-words-chars',  # Optional,
        params=params,  # Optional,
        tags=[
            model_tuner.hparams.coef, 
            'advprop' if model_tuner.hparams.advprop else 'imagenet',
            'expand_bifpn' if model_tuner.hparams.expand_bifpn else 'no_expand',
            'factor2' if model_tuner.hparams.factor2 else 'factor4',
            f'scale={data_module.scale}',
            f'repeat_bifpn={model_tuner.hparams.repeat_bifpn}',
            f'bifpn_channels={model_tuner.hparams.bifpn_channels}'
        ],  # Optional,
    )

    neptune_logger.experiment.log_artifact('hparams.yaml')

    # CALLBACK

    filepath = f'ckps/{neptune_logger.version}' + 'wc-effdet-{epoch}-{val_loss:.4f}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        mode='min',
        save_top_k=2
    )

    # TRAINER
    trainer = pl.Trainer(**hparams['trainer'], logger=neptune_logger, checkpoint_callback=checkpoint_callback)

    # FITTING
    trainer.fit(model_tuner, datamodule=data_module)
