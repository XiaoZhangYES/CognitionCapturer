from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.components.THINGSEEG_utils import EEGDataset, EEGDatasetDistributed, RouteDateset, VAEDataset


class EEGDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            EEGdata_path_250hz: str = "data/",
            EEGdata_path_200hz: str = "data/",
            EEGdata_path_100hz: str = "data/",
            image_datapath: str = "data/",
            feature_path: str = "data/",
            exclude_subject: str = None,
            use_ori_feature: bool = True,
            subjects: Tuple[str] = None,
            use_route: bool = False,
            ratio: float = None,
            ckpt_path: str = None,
            batch_size_val: int = None,
            batch_size: int = 512,
            use_vae: bool = False,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param ratio: data split's ratio, if ratio = 0, denotes all data are used for encoder's training
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.EEG_datapth = self.hparams.EEGdata_path_250hz

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_router: Optional[Dataset] = None
        self.use_vae = use_vae
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).

        Self's:
        note: this func only runs one time
        make all data_prepared, like preprocess module and downloading

        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = EEGDataset(EEGdata_path=self.EEG_datapth,
                                  imagedata_path=self.hparams.image_datapath,
                                  feature_path=self.hparams.feature_path,
                                  subjects=tuple(self.hparams.subjects),
                                  exclude_subject=self.hparams.exclude_subject,
                                  use_ori_feature=self.hparams.use_ori_feature,
                                  train=True,
                                  classes=None)
            route_size = int(self.hparams.ratio * len(trainset))
            train_size = len(trainset) - route_size
            train_dataset, route_dataset = random_split(trainset, [train_size, route_size], generator=torch.Generator())
            testset = EEGDataset(EEGdata_path=self.EEG_datapth,
                                 imagedata_path=self.hparams.image_datapath,
                                 feature_path=self.hparams.feature_path,
                                 subjects=tuple(self.hparams.subjects),
                                 exclude_subject=self.hparams.exclude_subject,
                                 use_ori_feature=self.hparams.use_ori_feature,
                                 train=False,
                                 classes=None)

            # dataset into vae if use_vae param setted
            if self.use_vae is True:
                train_dataset = VAEDataset(EEGdataset=train_dataset, mode='train')
                testset = VAEDataset(EEGdataset=testset, mode='test')

            self.data_router = route_dataset
            self.data_train = train_dataset
            self.data_val = testset
            self.data_test = testset

    def _get_route_dataset(self):
        """
        设置datarouter
        """
        self.data_router = RouteDateset(eegdataset=self.data_router,
                                    ckpt_path=self.hparams.ckpt_path)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.
        :return: The train dataloader.
        """
        if self.hparams.ckpt_path is not None:
            self._get_route_dataset()
        load_data = self.data_train if self.hparams.use_route is False else self.data_router
        return DataLoader(
            dataset=load_data,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=self.hparams.drop_last
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        if self.trainer.world_size > 1:
            # raise NotImplementedError("bug in multi GPU mode !")
            self.data_val = EEGDatasetDistributed(self.data_val)

        # define batchsize
        if self.hparams.batch_size_val is not None:
            bsz = self.hparams.batch_size_val
        else:
            bsz = 500
        return DataLoader(
            dataset=self.data_val,
            batch_size=bsz,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        if self.trainer.world_size > 1:
            self.data_test = EEGDatasetDistributed(self.data_test)

        # define batchsize
        if self.hparams.batch_size_val is not None:
            bsz = self.hparams.batch_size_val
        else:
            bsz = 500

        return DataLoader(
            dataset=self.data_test,
            batch_size=bsz,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = EEGDataModule()
