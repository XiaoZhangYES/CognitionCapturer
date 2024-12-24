"""
revised from https://github.com/dongyangli-del/EEG_Image_decode, thanks!
"""
import os
from typing import Any, Tuple
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split
import math
from deprecated import deprecated
from src.models.components.Cogcap.Cogcap import Cogcap, Proj_img, Proj_text, Proj_depth
from src.models.components.utils import load_model


class EEGDataset(data.Dataset):

    def __init__(self,
                 EEGdata_path: str = "data/",
                 imagedata_path: str = "data/image_set/",
                 subjects: Tuple[str, ...] = (
                         'sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09',
                         'sub-10'),
                 exclude_subject: str = None,
                 return_subjects: bool = False,
                 feature_path: str = "data/features/",
                 text_type: str = "BLIP2",
                 train: bool = True,
                 time_window: Tuple[float, float] = (0.0, 1.0),
                 classes: Tuple[int, ...] = None,
                 use_ori_feature: bool = True,
                 pictures: Any = None, ) -> None:
        """
        revise from "https://osf.io/w28xy", "https://osf.io/xp7c2"
        Image: The training images partition has 1,654 object concepts, with 10 images per concept
        (for a total of 16,540 image conditions),

        :param EEGdata_path: The EEG data directory
        :param imagedata_path: The image data directory
        :param exclude_subject: Whether to test on one subject (only works when dataset's subject is above 1)
        :param subjects: The subjects involved for training and tesing, if not assigned, default use all 10 subjects
                         e.g: subjects = ['sub-01', 'sub-02', ...]
        :param train: Train or Test datasets
        :param time_window: Time split the EEG signal used
        :param feature_path: Important for experiment, load features by the path
        :param text_type:Important for experiment (original | BLIP2)
        :param classes: If not none, select classed in classes
        :param pictures: 需要与classes一起送进，指定了classes中要传的图片索引
        in original code, the pictures denote which picture to load
        """

        self.EEGdata_path = EEGdata_path
        self.imagedata_path = imagedata_path
        self.return_subjects = return_subjects
        self.train = train
        self.subjects = tuple(os.listdir(EEGdata_path)) if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window

        if classes is not None:
            self.n_cls = len(classes)
        else:
            self.n_cls = 1654 if train else 200  # predefined in the paper

        self.samples_per_class = 10 if train else 1  # predefined in the paper
        self.text_type = text_type
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject

        # For divide time and channel, will be assigned in load_data function
        self.ch_names = None
        self.times = None

        # Try to load the saved features if they exist
        self.use_ori_feature = use_ori_feature
        self.train_status = "train" if self.train else "test"  # For divide features
        self.feature_path = feature_path

        # assert any subjects in subject_list
        assert any(sub in tuple(os.listdir(EEGdata_path)) for sub in self.subjects)

        self.data, self.labels, self.text, self.img, self.subject_labels = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        self.img_features, self.text_features, self.depth_features, self.aug_img_featrues = self.get_feature(
            use_ori_feature)

    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        if self.train:
            directory = os.path.join(self.imagedata_path, "image_set", "training_images")
        else:
            directory = os.path.join(self.imagedata_path, "image_set", "test_images")

        # dirname: list with 1654 classes
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        # select dirnames according to classes
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        ### Get texts ###

        if self.text_type == "Original":
            # dir: [num_classname]
            # CAUTION: there method's finally index in __getitem__ method is different
            for dir in dirnames:
                try:
                    idx = dir.index('_')
                    description = dir[idx + 1:]
                except ValueError:
                    print(f"Skipped: {dir} due to no '_' found.")
                    continue

                description = f"This picture is {description}"
                texts.append(description)

        elif self.text_type == "BLIP2":
            if self.train:
                texts_filename = os.path.join(self.imagedata_path, "image_set", "img_description",
                                              "texts_BLIP2_train.npy")
            else:
                texts_filename = os.path.join(self.imagedata_path, "image_set", "img_description",
                                              "texts_BLIP2_test.npy")
            if os.path.exists(texts_filename):
                texts_np = np.load(texts_filename)
                texts = texts_np.tolist()
            else:  # todo add text preprocess module if text not exit
                raise NotImplementedError
                # for _, img in enumerate(self.img):
                #     description = get_description(img)
                #     texts.append(description)

        ### Get texts ###

        ### Get images ###

        # Get image folders according to directory
        all_folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        all_folders.sort()

        if self.classes is not None:
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                folder = all_folders[class_idx]
                folder_path = os.path.join(directory, folder)
                all_images = [img for img in os.listdir(folder_path) if
                              img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                for idx in range(len(all_images)):
                    images.append(os.path.join(folder_path, all_images[idx]))
        elif self.classes is None:
            images = []
            for folder in all_folders:
                folder_path = os.path.join(directory, folder)
                all_images = [img for img in os.listdir(folder_path) if
                              img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            raise NotImplementedError

        ### Get images ###

        ### Get EEG ###

        times = None
        ch_names = None
        subject_labels = []
        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:
                    continue
                file_path = os.path.join(self.EEGdata_path, subject, 'preprocessed_eeg_training.npy')
                data = np.load(file_path, allow_pickle=True)
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()

                detach_index = 0 if "MVNN" in self.EEGdata_path else \
                    (torch.from_numpy(data['times']) == 0).nonzero().item()  # find EEG data's start index
                times = torch.from_numpy(data['times']).detach()[detach_index:]  # Then detach

                subject_label = torch.full((len(data['preprocessed_eeg_data']),),
                                           int(subject[4:]),
                                           dtype=torch.long).detach()

                # MVNN's time is from (-0.2 - +0.8), so += 0.2 for consistency
                if "MVNN" in self.EEGdata_path:
                    times += 0.2

                ch_names = data['ch_names']

                if self.classes is not None and self.pictures is not None: # choose pictures only
                    raise NotImplementedError('needs to write subject_labels code')
                    # for c, p in zip(self.classes, self.pictures):
                    #     start_index = c * 1 + p
                    #     if start_index < len(preprocessed_eeg_data):
                    #         preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + 1]
                    #         labels = torch.full((1,), c, dtype=torch.long).detach()
                    #         data_list.append(preprocessed_eeg_data_class)
                    #         label_list.append(labels)

                elif self.classes is not None and self.pictures is None: # choose classes only
                    raise NotImplementedError('needs to write subject_labels code')
                    # for c in self.classes:
                    #     start_index = c * self.samples_per_class
                    #     preprocessed_eeg_data_class = preprocessed_eeg_data[
                    #                                   start_index: start_index + self.samples_per_class]
                    #     labels = torch.full((self.samples_per_class,), c, dtype=torch.long).detach()
                    #     data_list.append(preprocessed_eeg_data_class)
                    #     label_list.append(labels)
                else:
                    for i in range(self.n_cls):
                        start_index = i * self.samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                                                      start_index: start_index + self.samples_per_class]
                        labels = torch.full((self.samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                    subject_label = subject_label.repeat(4) # 4 is calculated
                    subject_labels.append(subject_label)

            else:  # test data
                if subject == self.exclude_subject or self.exclude_subject is None:
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.EEGdata_path, subject, file_name)
                    data = dict(np.load(file_path, allow_pickle=True).items())
                    subject_label = torch.full((len(data['preprocessed_eeg_data']),),
                                                int(subject[4:]),
                                                dtype=torch.long).detach()
                    subject_labels.append(subject_label)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()

                    detach_index = 0 if "MVNN" in self.EEGdata_path else (
                            torch.from_numpy(data['times']) == 0).nonzero().item()
                    times = torch.from_numpy(data['times']).detach()[detach_index:]
                    if "MVNN" in self.EEGdata_path:
                        times += 0.2
                    ch_names = data['ch_names']

                    for i in range(self.n_cls):
                        ### NOTE: 取平均也行？？？ ###
                        if self.classes is not None and i not in self.classes:
                            # If we've defined specific classes and the current class is not in the list, skip
                            continue
                        start_index = i * self.samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                                                      start_index:start_index + self.samples_per_class]
                        labels = torch.full((self.samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  # Add labels to the label list
                else:
                    continue

        # load EEGdata over

        # data_list: (image, see times, channels, seqlength) ???
        data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:]) if self.train else \
            torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)

        label_tensor = torch.cat(label_list, dim=0)
        if self.train:
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)
                mapping = {val.item(): index for index, val in enumerate(unique_values)}
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
        else:
            pass

        if times is None or ch_names is None:
            raise ValueError("Times and channel names must be provided.")

        self.times = times
        self.ch_names = ch_names

        ### Get EEG ###

        subject_labels = torch.cat(subject_labels, dim=0)

        return data_tensor, label_tensor, texts, images, subject_labels

    def extract_eeg(self, eeg_data, time_window):
        # Get the indices of the times within the specified window
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]  # 200hz eeg:(200)
        return extracted_data

    def get_feature(self, use_ori_feature):

        if use_ori_feature:  # use dongyangli-del's features
            raise NotImplementedError("Do with depth aug img_features")  # TODO here not implemented

            # self.feature_path = os.path.join(f"{self.feature_path}clip_BLIP2_features_{self.train_status}.pt")
            # if not os.path.exists(self.feature_path):
            #     raise ValueError(f'No feature file found in {self.feature_path}')
            # image_features = torch.load(self.feature_path, weights_only=True)['img_features']
            # text_features = torch.load(self.feature_path, weights_only=True)['text_features']
            # return image_features, text_features, image_features, text_features
        else:
            test_str = "_test"
            image_features = os.path.join(self.feature_path,
                                          f"image_original_features_clip_dict{test_str if not self.train else ''}.pt")
            text_features = os.path.join(self.feature_path,
                                         f"text_finegrain_features_clip_dict{test_str if not self.train else ''}.pt")
            depth_features = os.path.join(self.feature_path,
                                          f"image_depth_features_clip_dict{test_str if not self.train else ''}.pt")
            aug_image_features = os.path.join(self.feature_path,
                                              f"image_aug_features_clip_dict{test_str if not self.train else ''}.pt")
            image_features = torch.load(image_features, weights_only=True)
            text_features = torch.load(text_features, weights_only=True)
            depth_features = torch.load(depth_features, weights_only=True)
            aug_image_features = torch.load(aug_image_features, weights_only=True)

            return image_features, text_features, depth_features, aug_image_features

    def get_index(self, index):
        # important! this variable notes how to split test dataset
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * self.samples_per_class
            else:
                index_n_sub_test = len(self.classes) * 1 * self.samples_per_class
                index_n_sub_train = len(self.classes) * 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test) // (1 * self.samples_per_class)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // 4
            else:
                img_index = (index % index_n_sub_test) // self.samples_per_class
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * self.samples_per_class
            else:
                index_n_sub_test = len(self.classes) * 1 * self.samples_per_class
                index_n_sub_train = len(self.classes) * 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test) // (1 * self.samples_per_class)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // 4
            else:
                img_index = (index % index_n_sub_test) // self.samples_per_class
        return text_index, img_index
    def __getitem__(self, index):
        """
        Get the data and label corresponding to "index"
        index: (subjects * classes * 10 * 4)
        Deleted original text and image
        """

        x = self.data[index]
        label = self.labels[index]
        subject_label = self.subject_labels[index]
        text_index, img_index = self.get_index(index)
        text = self.text[text_index]
        img = self.img[img_index]

        img_name = os.path.basename(img)  # For index

        depth_img_name = img_name.replace("jpg", "png")  # little bug

        if self.use_ori_feature: # ori's index are aligned with eeg
            text_features = self.text_features[text_index]
            img_features = self.img_features[img_index]
            depth_features = self.depth_features[img_index]
            aug_img_features = self.aug_img_featrues[img_index]
        else: # new features are aligned using name
            text_features = self.text_features[img_name].squeeze(0)
            img_features = self.img_features[img_name].squeeze(0)
            depth_features = self.depth_features[depth_img_name].squeeze(0)
            aug_img_features = self.aug_img_featrues[img_name].squeeze(0)

        if self.return_subjects is True:
            return x, label, text, text_features, img, img_features, depth_features, aug_img_features, img_index, index, subject_label
        return x, label, text, text_features, img, img_features, depth_features, aug_img_features, img_index, index
    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same


class EEGDatasetDistributed(Dataset):

    def __init__(self, EEGDataset) -> None:
        self.EEGDataset = EEGDataset
        self.length = len(EEGDataset) * 2

    def __getitem__(self, index):
        actual_index = int(index / 2)
        return self.EEGDataset[actual_index]
    def __len__(self):
        return self.length



def openimg(path: str):
    img = Image.open(path)
    plt.imshow(img)
    plt.show()


def dict_feature_2_no_dict(train=False):
    for content in ['image_aug']:
        modality_features = []
        modality_features_dict = torch.load(
            f"/data/zkf/ModelWeights/Things_dataset"
            f"/model_pretrained/data_features/{content}_features_clip_dict{'_test' if not train else ''}.pt",
            weights_only=True)
        for index, key in enumerate(modality_features_dict):
            modality_features.append(modality_features_dict[key])
        modality_features = torch.stack(modality_features, dim=1).squeeze(0)
        torch.save(modality_features,
                   f"/data/zkf/ModelWeights/Things_dataset/model_pretrained/data_features/{content}_features_clip{'_test' if not train else ''}.pt")


def plt_eeg(eeg):
    """

    """
    import matplotlib.pyplot as plt
    x = np.arange(len(eeg))

    # 绘制线图
    plt.plot(x, eeg)
    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.title('Sinusoidal Wave')
    plt.grid(True)
    plt.show()


def get_cos_metric(img_features: torch.Tensor,
                   text_features: torch.Tensor,
                   depth_features: torch.Tensor,
                   aug_img_features: torch.Tensor) -> dict:
    """
    Calculate the cosine similarity between the image, text, and depth features.
    """
    output = {"img * text": cosine_similarity(img_features, text_features, dim=0),
              "img * depth": cosine_similarity(img_features, depth_features, dim=0),
              "img * aug_img": cosine_similarity(img_features, aug_img_features, dim=0),
              "text * depth": cosine_similarity(text_features, depth_features, dim=0),
              "text * aug_img": cosine_similarity(text_features, aug_img_features, dim=0),
              "depth * aug_img": cosine_similarity(depth_features, aug_img_features, dim=0)}

    return output


def ori_test():
    data_path_250hz = "/HDD2/Things_dataset/Things_eeg/preprocessed_data_250hz/"
    image_datapath = "/HDD2/Things_dataset/Things_eeg/"
    train_dataset = EEGDataset(EEGdata_path=data_path_250hz,
                               imagedata_path=image_datapath,
                               feature_path="/HDD2/Things_dataset/model_pretrained/data_features/",
                               subjects=('sub-01',),
                               exclude_subject=None,
                               use_ori_feature=False,
                               train=True,
                               classes=None)
    dict_feature_2_no_dict(train=True)
    test_dataset = EEGDataset(EEGdata_path=data_path_250hz,
                              feature_path="/HDD2/Things_dataset/model_pretrained/data_features/",
                              subjects=('sub-01',),
                              imagedata_path=image_datapath,
                              exclude_subject="sub-01",
                              use_ori_feature=False,
                              train=False,
                              classes=None)

    x, label, text, text_features, img, img_features, depth_features, aug_img_features, img_index = train_dataset[54]
    dict = get_cos_metric(img_features, text_features, depth_features, aug_img_features)
    print(dict)
    openimg(img)
    x, label, text, text_features, img, img_features, depth_features, aug_img_features, img_index = test_dataset[20]
    dict = get_cos_metric(img_features, text_features, depth_features, aug_img_features)
    print(dict)
    print(x.shape, label, text, text_features.shape, img, img_features.shape, depth_features.shape)
    openimg(img)


def route_dataset_test():
    data_path_250hz = "/data/zkf/ModelWeights/Things_dataset/Things_eeg/preprocessed_data_250hz/"
    image_datapath = "/data/zkf/ModelWeights/Things_dataset/Things_eeg/"
    trainset = EEGDataset(EEGdata_path=data_path_250hz,
                          imagedata_path=image_datapath,
                          feature_path="/data/zkf/ModelWeights/Things_dataset/model_pretrained/data_features/",
                          subjects=('sub-01', 'sub-02'),
                          exclude_subject=None,
                          use_ori_feature=False,
                          train=True,
                          )
    batch = trainset[0]
    plt_eeg(np.array(batch[0][1]))


if __name__ == '__main__':
    route_dataset_test()
