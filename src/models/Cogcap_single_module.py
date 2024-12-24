from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule
from lavis.models.clip_models.loss import ClipLoss
from src.models.components.Cogcap.Cogcap_eval import Top_K_Accuracy


class Cogcap_Module(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            eegnet: torch.nn.Module,
            imgnet: torch.nn.Module,
            textnet: torch.nn.Module,
            depthnet: torch.nn.Module,
            augnet: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_type: str,
            feature_path: str,
            compile: bool,
            modality: str
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param eegnet: The model to train.
        :param imgnet: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.EEGmodel = eegnet
        self.imagemodel = imgnet
        self.textmodel = textnet
        self.depthmodel = depthnet
        self.aug_img_model = augnet

        # features for calculating loss
        self.image_features_all, self.text_features_all, self.depth_features_all, self.aug_img_features_all = \
            self.get_feature_all(self.hparams.feature_path)

        # loss function base
        self.criterion = ClipLoss()

        # softmax and t already in here
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).requires_grad_(False)

        self.top200_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=200, modality=self.hparams.modality)
        self.top100_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=100, modality=self.hparams.modality)
        self.top50_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=50, modality=self.hparams.modality)

    def forward(self,
                x: torch.Tensor,
                image_feature: torch.Tensor,
                text_feature: torch.Tensor,
                depth_feature: torch.Tensor,
                aug_image_feature: torch.Tensor,) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.


        """
        ### For original paper experiment code ###
        # return self.EEGmodel(x), self.imagemodel(image_feature), \
        #     text_feature, depth_feature, \
        #     aug_image_feature

        ### Used all projection model code ###
        return self.EEGmodel(x), self.imagemodel(image_feature), \
            self.textmodel(text_feature), self.depthmodel(depth_feature), \
            self.aug_img_model(aug_image_feature)



    def loss_cal(self, eeg_features, image_features, text_features, depth_features):
        """
        decide loss's type according to self.hparams.loss_type, then calculate
        :return: loss's num
        """
        if self.hparams.loss_type == "one modality":
            if self.hparams.modality == "image":
                selected_features = image_features
            elif self.hparams.modality == "text":
                selected_features = text_features
            elif self.hparams.modality == "depth":
                selected_features = depth_features
            else:
                raise NotImplementedError(f"Modality False! {self.hparams.modality}")
            return self.criterion(eeg_features, selected_features, self.logit_scale)

        if self.hparams.loss_type == "original":
            return self.criterion(eeg_features, image_features, self.logit_scale) + \
                0.01 * self.criterion(eeg_features, text_features, self.logit_scale)

        if self.hparams.loss_type == "img_text_loss_best":
            return self.criterion(eeg_features, image_features, self.logit_scale) + \
                0.1 * self.criterion(eeg_features, text_features, self.logit_scale)

        if self.hparams.loss_type == "depth_img":
            return self.criterion(eeg_features, depth_features, self.logit_scale) + \
                0.1 * self.criterion(eeg_features, image_features, self.logit_scale)

        if self.hparams.loss_type == "add_all":
            return self.criterion(eeg_features, text_features) + \
                0.1 * self.criterion(eeg_features, image_features) + \
                0.01 * self.criterion(eeg_features, depth_features)

        raise NotImplementedError(f"please select a appropriate loss type! now: {self.hparams.loss_type}")

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

    def get_feature_all(self, feature_path) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return:
        features: (dict) "image's name": torch.Tensor
        """

        ### THIS IS OLD ###
        # Copied from THINGSEEG_utils.py and revised
        # feature_path = [os.path.join(f"{feature_path}clip_BLIP2_features_train.pt"),
        #                 os.path.join(f"{feature_path}clip_BLIP2_features_test.pt")]
        # image_features, text_features = [], []
        # for path in feature_path:
        #     if not os.path.exists(path):
        #         raise ValueError(f'No feature file found in {path}')
        #     saved_features = torch.load(path)
        #     text_features.append(saved_features['text_features'])
        #     image_features.append(saved_features['img_features'])
        # image_features = image_features[1]
        # text_features = text_features[1]
        ### THIS IS OLD ###

        ### There features are original file which is dict ###
        # image_features = torch.load(
        #     f"/HDD2/Things_dataset/model_pretrained/data_features/image_original_features_clip_dict_test.pt")
        # text_features = torch.load(
        #     f"/HDD2/Things_dataset/model_pretrained/data_features/text_finegrain_features_clip_dict_test.pt")
        # depth_features = torch.load(
        #     f"/HDD2/Things_dataset/model_pretrained/data_features/image_depth_features_clip_dict_test.pt")
        # aug_image_features = torch.load(
        #     f"/HDD2/Things_dataset/model_pretrained/data_features/image_aug_features_clip_dict_test.pt")
        ### There features are original file which is dict ###

        ### There features are inherited from ..._features_dict_test.pt file ###
        image_features = torch.load(
            f"/HDD2/Things_dataset/model_pretrained/data_features/features_for_eval/image_features_clip_test.pt")
        text_features = torch.load(
            f"/HDD2/Things_dataset/model_pretrained/data_features/features_for_eval/text_features_clip_test.pt")
        depth_features = torch.load(
            f"/HDD2/Things_dataset/model_pretrained/data_features/features_for_eval/depth_features_clip_test.pt")
        aug_image_features = torch.load(
            f"/HDD2/Things_dataset/model_pretrained/data_features/features_for_eval/aug_img_features_clip_test.pt")
        ### There features is inherited from ..._features_dict_test.pt file ###

        return image_features, text_features, depth_features, aug_image_features

    def model_train_step(
        self,
        batch: Tuple[Any, ...], #todo
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        eeg_data, label, text, text_features, img, image_features, depth_features, aug_img_features, index = batch



        eeg_features, image_features, text_features, depth_features, aug_img_features = \
            self.forward(x=eeg_data,
                         image_feature=image_features,
                         text_feature=text_features,
                         depth_feature=depth_features,
                         aug_image_feature=aug_img_features)
        loss = self.loss_cal(eeg_features=eeg_features,
                             image_features=image_features,
                             text_features=text_features,
                             depth_features=depth_features)
        return loss, eeg_features, label, text_features, image_features, depth_features, aug_img_features

    def model_test_step(
        self,
        batch: Tuple[Any, ...], #todo
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        eeg_data, label, text, text_features, img, image_features, depth_features, aug_img_features, index = batch

        # Note: if you want to save features, add save_features func here
        # save_features(img, self.image_features_all, "image_features_clip_test.pt")
        # save_features(img, self.text_features_all, "text_features_clip_test.pt")
        # save_features(img, self.depth_features_all, "depth_features_clip_test.pt")
        # save_features(img, self.aug_img_features_all, "aug_img_features_clip_test.pt")

        eeg_features, image_features, text_features, depth_features, aug_img_features = \
            self.forward(x=eeg_data,
                         image_feature=image_features,
                         text_feature=text_features,
                         depth_feature=depth_features,
                         aug_image_feature=aug_img_features)
        loss = self.loss_cal(eeg_features=eeg_features,
                             image_features=image_features,
                             text_features=text_features,
                             depth_features=depth_features)
        return loss, eeg_features, label, text_features, image_features, depth_features, aug_img_features

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, eeg_features, label, text_features, image_features, depth_features, aug_img_features = self.model_train_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, eeg_features, label, text_features, image_features, depth_features, aug_img_features = self.model_test_step(batch)

        #eval
        top200class_accuracy, top200class_top5_acc = self.top200_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                                                    targets=label,
                                                                                                    img_features_all=self.image_features_all,
                                                                                                    text_features_all=self.text_features_all,
                                                                                                    depth_features_all=self.depth_features_all,
                                                                                                    aug_img_features_all=self.aug_img_features_all)
        top100class_accuracy, top100class_top5_acc = self.top100_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                                                    targets=label,
                                                                                                    img_features_all=self.image_features_all,
                                                                                                    text_features_all=self.text_features_all,
                                                                                                    depth_features_all=self.depth_features_all,
                                                                                                    aug_img_features_all=self.aug_img_features_all)
        top50class_accuracy, top50class_top5_acc = self.top50_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                                                 targets=label,
                                                                                                 img_features_all=self.image_features_all,
                                                                                                 text_features_all=self.text_features_all,
                                                                                                 depth_features_all=self.depth_features_all,
                                                                                                 aug_img_features_all=self.aug_img_features_all)
        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy", top200class_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc", top200class_top5_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy", top100class_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc", top100class_top5_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy", top50class_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc", top50class_top5_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, eeg_features, label, text_features, image_features, depth_features, aug_img_features = self.model_test_step(batch)


        # eval
        top200class_accuracy, top200class_top5_acc = self.top200_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                                                    targets=label,
                                                                                                    img_features_all=self.image_features_all,
                                                                                                    text_features_all=self.text_features_all,
                                                                                                    depth_features_all=self.depth_features_all,
                                                                                                    aug_img_features_all=self.aug_img_features_all)
        top100class_accuracy, top100class_top5_acc = self.top100_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                                                    targets=label,
                                                                                                    img_features_all=self.image_features_all,
                                                                                                    text_features_all=self.text_features_all,
                                                                                                    depth_features_all=self.depth_features_all,
                                                                                                    aug_img_features_all=self.aug_img_features_all)
        top50class_accuracy, top50class_top5_acc = self.top50_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                                                 targets=label,
                                                                                                 img_features_all=self.image_features_all,
                                                                                                 text_features_all=self.text_features_all,
                                                                                                 depth_features_all=self.depth_features_all,
                                                                                                 aug_img_features_all=self.aug_img_features_all)
        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_accuracy", top200class_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top200class_top5_acc", top200class_top5_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_accuracy", top100class_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top100class_top5_acc", top100class_top5_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_accuracy", top50class_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top50class_top5_acc", top50class_top5_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        ### COMPILE ABANDONED ###
        # if self.hparams.compile: #  and stage == "fit" why
        #     self.EEGmodel = torch.compile(self.EEGmodel)
        #     self.imagemodel = torch.compile(self.imagemodel)
        ### COMPILE ABANDONED ###



    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Cogcap_Module(None, None, None, None, None, None, None, None,
                       "/HDD2/Things_dataset/model_pretrained/data_features/",
                      None, None)
