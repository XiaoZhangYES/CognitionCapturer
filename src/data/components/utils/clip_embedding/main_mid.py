'''
this code contains dataset's define and change
todo:IMPORTANT:samples_per_class needs to adjust
'''

import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import open_clip
import os
import json
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer



class FeatureGet:
    def __init__(self):
        self.device = "cuda:1"
        self.vlmodel, self.preprocess_train = self._get_model()

    def _get_model(self):
        vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='/HDD2/Things_dataset/model_pretrained/clip/open_clip_pytorch_model.bin',
            precision='fp32',
            device=self.device)
        return vlmodel, preprocess_train

    def _Textencoder(self, text):
        text_input = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_feature = self.vlmodel.encode_text(text_input)
            text_feature = F.normalize(text_feature, dim=-1).detach()
        return text_feature


    def _ImageEncoder(self, image):
        img_input = self.preprocess_train(Image.open(image).convert("RGB")).to(self.device).unsqueeze(0)
        with torch.no_grad():
            image_feature = self.vlmodel.encode_image(img_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
        return image_feature

    def get_text_feature(self, type):
        saved_text_path = "/HDD2/Things_dataset/Things_eeg/image_set/img_description/"
        saved_img_path = "/HDD2/Things_dataset/Things_eeg/image_set/img_path/"
        with torch.no_grad():
            # 'train.npy' , 'test.npy'
            for text_file_name, img_file_name in zip(['texts_BLIP2_train.npy'], ['train_image.npy']):
                text_features_dict = {}
                textfile_name = os.path.join(saved_text_path, text_file_name)
                img_file_name = os.path.join(saved_img_path, img_file_name)
                text_path = np.load(textfile_name).tolist()
                img_path = np.load(img_file_name).tolist()
                for j, text_description in enumerate(text_path):

                    print(f"{j + 1} / {len(text_path)}")

                    if type != 'finegrain':
                        raise NotImplementedError("todo")

                    ### model forward ###
                    output = self._Textencoder(text_description)

                    ### to cpu and append to list, then empty cache ###
                    output = output.cpu()
                    text_features_dict[os.path.basename(img_path[j])] = output
                    torch.cuda.empty_cache()

                torch.save(text_features_dict,
                           f"/HDD2/Things_dataset/model_pretrained/data_features/text_{type}_features_clip_dict.pt")
    def get_feature(self, type):
        saved_path = "/HDD2/Things_dataset/Things_eeg/image_set/img_path/"
        with torch.no_grad():
            # , 'test_image.npy'
            for i, file_name in enumerate(['train_image.npy']):
                image_features_dict = {}
                file_name = os.path.join(saved_path, file_name)
                path = np.load(file_name).tolist()
                for j, img_path in enumerate(path):

                    print(f"{j + 1} / {len(path)}")

                    if type == 'depth':
                        ### img_path change into depth image's pth ###
                        img_path = img_path.replace("image_set", "image_depth_set").replace(".jpg", ".png")
                    if type == 'aug':
                        img_path = img_path.replace("training_images", "aug_images")

                    ### model forward ###
                    output = self._ImageEncoder(img_path)

                    ### to cpu and append to list, then empty cache ###
                    output = output.cpu()
                    image_features_dict[os.path.basename(img_path)] = output
                    torch.cuda.empty_cache()

                torch.save(image_features_dict,
                           f"/HDD2/Things_dataset/model_pretrained/data_features/image_{type}_features_clip_dict.pt")
def open_img(img_path):
    '''
    input : img path
    output : img with plt showed
    '''
    img = Image.open(img_path)
    plt.figure(f"{img_path}")
    plt.imshow(img)
    plt.axis('on')
    plt.title('image')
    plt.show()


if __name__ == "__main__":
    instance = FeatureGet()
    instance.get_text_feature('finegrain')






