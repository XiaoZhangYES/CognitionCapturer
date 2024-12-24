import statistics
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from src.models.components.Cogcap.loss import SupConLoss
from src.models.components.Cogcap.cliploss import ClipLoss
from lightning import LightningModule
import torch.nn.functional as F
from torch import cosine_similarity

from src.models.components.Cogcap.Cogcap_eval import Top_K_Accuracy


class Cogcap_allmodalitymodule(LightningModule):

    def __init__(
            self,
            eegnet_img: torch.nn.Module,
            eegnet_text: torch.nn.Module,
            eegnet_depth: torch.nn.Module,
            imgnet: torch.nn.Module,
            textnet: torch.nn.Module,
            depthnet: torch.nn.Module,
            automatic_optimization: bool,
            loss_type: str,
            top_k: int,
            cos_batch: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            feature_path: str,
            compile: bool,
    ) -> None:
        super().__init__()

        self.automatic_optimization = automatic_optimization
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.EEGmodel_img = eegnet_img
        self.EEGmodel_text = eegnet_text
        self.EEGmodel_depth = eegnet_depth
        self.imagemodel = imgnet
        self.textmodel = textnet
        self.depthmodel = depthnet
        self.loss_type = loss_type

        # loss function base
        self.criterion = self._select_loss()

        # softmax and t already in here
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).requires_grad_(False)

        self.top200_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=200)
        self.top100_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=100)
        self.top50_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=50)

    def _select_loss(self):
        if self.loss_type == "Original":
            return ClipLoss(top_k=self.hparams.top_k, cos_batch=self.hparams.cos_batch)
        elif self.loss_type == "Supervised":
            return SupConLoss(temperature=0.1)
        else:
            raise NotImplementedError("Set loss type correctly!")

    def forward(self,
                x: torch.Tensor,
                modality_feature: torch.Tensor,
                eegmodel: torch.nn.Module,
                modalitymodel: torch.nn.Module
                ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:

        return eegmodel(x), modalitymodel(modality_feature)

    def loss_cal(self, eeg_features, modality_features, img_index):
        """
        decide loss's type according to self.hparams.loss_type, then calculate
        :return: loss
        """
        if self.loss_type == "Supervised":
            input = torch.stack((eeg_features, modality_features), dim=1)
            # L2 Normalize following
            # https://github.com/HobbitLong/SupContrast
            input = F.normalize(input, dim=2)
            return self.criterion(input, img_index)
        elif self.loss_type == "Original":
            return self.criterion(eeg_features, modality_features, self.logit_scale, img_index)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

    def model_train_step(
            self,
            batch: Tuple[Any, ...],  # todo
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        eeg_data, label, text, text_features, img, image_features, depth_features, aug_img_features, img_index, index = batch

        eeg_features_img, image_features = self.forward(x=eeg_data,
                                                        modality_feature=image_features,
                                                        eegmodel=self.EEGmodel_img,
                                                        modalitymodel=self.imagemodel)
        loss_img = self.loss_cal(eeg_features=eeg_features_img,
                                 modality_features=image_features,
                                 img_index=img_index)
        eeg_features_text, text_features = self.forward(x=eeg_data,
                                                        modality_feature=text_features,
                                                        eegmodel=self.EEGmodel_text,
                                                        modalitymodel=self.textmodel)
        loss_text = self.loss_cal(eeg_features=eeg_features_text,
                                  modality_features=text_features,
                                  img_index=img_index)
        eeg_features_depth, depth_features = self.forward(x=eeg_data,
                                                          modality_feature=depth_features,
                                                          eegmodel=self.EEGmodel_depth,
                                                          modalitymodel=self.depthmodel)
        loss_depth = self.loss_cal(eeg_features=eeg_features_depth,
                                   modality_features=depth_features,
                                   img_index=img_index)

        return loss_img, loss_text, loss_depth

    def model_test_step(
            self,
            batch: Tuple[Any, ...],  # todo
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        eeg_data, label, text, text_features, img, image_features, depth_features, aug_img_features, img_index, index = batch

        eeg_features_img, image_features = self.forward(x=eeg_data,
                                                        modality_feature=image_features,
                                                        eegmodel=self.EEGmodel_img,
                                                        modalitymodel=self.imagemodel)
        loss_img = self.loss_cal(eeg_features=eeg_features_img,
                                 modality_features=image_features,
                                 img_index=img_index)
        eeg_features_text, text_features = self.forward(x=eeg_data,
                                                        modality_feature=text_features,
                                                        eegmodel=self.EEGmodel_text,
                                                        modalitymodel=self.textmodel)
        loss_text = self.loss_cal(eeg_features=eeg_features_text,
                                  modality_features=text_features,
                                  img_index=img_index)
        eeg_features_depth, depth_features = self.forward(x=eeg_data,
                                                          modality_feature=depth_features,
                                                          eegmodel=self.EEGmodel_depth,
                                                          modalitymodel=self.depthmodel)
        loss_depth = self.loss_cal(eeg_features=eeg_features_depth,
                                   modality_features=depth_features,
                                   img_index=img_index)
        return loss_img, loss_text, loss_depth, eeg_features_img, eeg_features_text, eeg_features_depth, label, text_features, image_features, depth_features

    def training_step(
            self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        img_opt, text_opt, depth_opt = self.optimizers()
        # batch_size = batch[0].shape[0]
        eeg_data, label, text, text_features, img, image_features, depth_features, aug_img_features, img_index, index = batch

        eeg_features_img, image_features = self.forward(x=eeg_data,
                                                        modality_feature=image_features,
                                                        eegmodel=self.EEGmodel_img,
                                                        modalitymodel=self.imagemodel)
        loss_img = self.loss_cal(eeg_features=eeg_features_img,
                                 modality_features=image_features,
                                 img_index=img_index)
        ### manual optimize ###
        img_opt.zero_grad()
        self.manual_backward(loss_img)
        img_opt.step()
        ### manual optimize ###

        eeg_features_text, text_features = self.forward(x=eeg_data,
                                                        modality_feature=text_features,
                                                        eegmodel=self.EEGmodel_text,
                                                        modalitymodel=self.textmodel)
        loss_text = self.loss_cal(eeg_features=eeg_features_text,
                                  modality_features=text_features,
                                  img_index=img_index)
        ### manual optimize ###
        text_opt.zero_grad()
        self.manual_backward(loss_text)
        text_opt.step()
        ### manual optimize ###

        eeg_features_depth, depth_features = self.forward(x=eeg_data,
                                                          modality_feature=depth_features,
                                                          eegmodel=self.EEGmodel_depth,
                                                          modalitymodel=self.depthmodel)
        loss_depth = self.loss_cal(eeg_features=eeg_features_depth,
                                   modality_features=depth_features,
                                   img_index=img_index)
        ### manual optimize ###
        depth_opt.zero_grad()
        self.manual_backward(loss_depth)
        depth_opt.step()
        ### manual optimize ###


        # todo 取消掉subset的使用
        self.log("loss_sum", loss_img + loss_text + loss_depth, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_text", loss_text, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_depth", loss_depth, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends.
        """
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_img, loss_text, loss_depth, eeg_features_img, eeg_features_text, eeg_features_depth, \
            label, text_features, image_features, depth_features = self.model_test_step(batch)

        top200class_accuracy, top200class_top5_acc = self.top200_acc_test.calculate_allmodality(
            EEG_img_features=eeg_features_img,
            EEG_text_features=eeg_features_text,
            EEG_depth_features=eeg_features_depth,
            img_features_all=image_features,
            text_features_all=text_features,
            depth_features_all=depth_features,
            seperate_calculate=True,
            targets=label
        )
        top100class_accuracy, top100class_top5_acc = self.top100_acc_test.calculate_allmodality(
            EEG_img_features=eeg_features_img,
            EEG_text_features=eeg_features_text,
            EEG_depth_features=eeg_features_depth,
            img_features_all=image_features,
            text_features_all=text_features,
            depth_features_all=depth_features,
            seperate_calculate=True,
            targets=label)
        top50class_accuracy, top50class_top5_acc = self.top50_acc_test.calculate_allmodality(
            EEG_img_features=eeg_features_img,
            EEG_text_features=eeg_features_text,
            EEG_depth_features=eeg_features_depth,
            img_features_all=image_features,
            text_features_all=text_features,
            depth_features_all=depth_features,
            seperate_calculate=True,
            targets=label)

        self.log("loss_sum", loss_img + loss_text + loss_depth, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_text", loss_text, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_depth", loss_depth, on_step=False, on_epoch=True, prog_bar=True)

        self.log("top200class_accuracy/image", top200class_accuracy[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy/text", top200class_accuracy[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy/depth", top200class_accuracy[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy/all", top200class_accuracy[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top200class_top5_acc/image", top200class_top5_acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc/text", top200class_top5_acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc/depth", top200class_top5_acc[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc/all", top200class_top5_acc[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top100class_accuracy/image", top100class_accuracy[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy/text", top100class_accuracy[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy/depth", top100class_accuracy[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy/all", top100class_accuracy[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top100class_top5_acc/image", top100class_top5_acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc/text", top100class_top5_acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc/depth", top100class_top5_acc[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc/all", top100class_top5_acc[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top50class_accuracy/image", top50class_accuracy[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy/text", top50class_accuracy[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy/depth", top50class_accuracy[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy/all", top50class_accuracy[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top50class_top5_acc/image", top50class_top5_acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc/text", top50class_top5_acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc/depth", top50class_top5_acc[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc/all", top50class_top5_acc[3], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_img, loss_text, loss_depth, eeg_features_img, eeg_features_text, eeg_features_depth, \
            label, text_features, image_features, depth_features = self.model_test_step(batch)

        # eval
        top200class_accuracy, top200class_top5_acc = self.top200_acc_test.calculate_allmodality(
            EEG_img_features=eeg_features_img,
            EEG_text_features=eeg_features_text,
            EEG_depth_features=eeg_features_depth,
            img_features_all=image_features,
            text_features_all=text_features,
            depth_features_all=depth_features,
            seperate_calculate=True,
            targets=label
        )
        top100class_accuracy, top100class_top5_acc = self.top100_acc_test.calculate_allmodality(
            EEG_img_features=eeg_features_img,
            EEG_text_features=eeg_features_text,
            EEG_depth_features=eeg_features_depth,
            img_features_all=image_features,
            text_features_all=text_features,
            depth_features_all=depth_features,
            seperate_calculate=True,
            targets=label)
        top50class_accuracy, top50class_top5_acc = self.top50_acc_test.calculate_allmodality(
            EEG_img_features=eeg_features_img,
            EEG_text_features=eeg_features_text,
            EEG_depth_features=eeg_features_depth,
            img_features_all=image_features,
            text_features_all=text_features,
            depth_features_all=depth_features,
            seperate_calculate=True,
            targets=label)

        self.log("loss_sum", loss_img + loss_text + loss_depth, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_img", loss_img, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_text", loss_text, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_depth", loss_depth, on_step=False, on_epoch=True, prog_bar=True)

        self.log("top200class_accuracy/image", top200class_accuracy[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy/text", top200class_accuracy[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy/depth", top200class_accuracy[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy/all", top200class_accuracy[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top200class_top5_acc/image", top200class_top5_acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc/text", top200class_top5_acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc/depth", top200class_top5_acc[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc/all", top200class_top5_acc[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top100class_accuracy/image", top100class_accuracy[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy/text", top100class_accuracy[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy/depth", top100class_accuracy[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy/all", top100class_accuracy[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top100class_top5_acc/image", top100class_top5_acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc/text", top100class_top5_acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc/depth", top100class_top5_acc[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc/all", top100class_top5_acc[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top50class_accuracy/image", top50class_accuracy[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy/text", top50class_accuracy[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy/depth", top50class_accuracy[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy/all", top50class_accuracy[3], on_step=False, on_epoch=True, prog_bar=True)

        self.log("top50class_top5_acc/image", top50class_top5_acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc/text", top50class_top5_acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc/depth", top50class_top5_acc[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc/all", top50class_top5_acc[3], on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_group = []
        for i in range(3):
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
            if self.hparams.scheduler is not None:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                optimizer_group.append({
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "train/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                })
            optimizer_group.append({"optimizer": optimizer})
        return optimizer_group[0], optimizer_group[1], optimizer_group[2]


if __name__ == "__main__":
    input = torch.randn(1, 63, 250)
