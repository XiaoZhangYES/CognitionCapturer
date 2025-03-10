import os
import re
from typing import Any, Tuple
import glob
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



def get_description(img):
    """
    根据输入的图像文件生成描述文本。
    使用预训练的Blip2模型来从图像中生成描述。如果GPU可用，则利用GPU加速模型推理。
    参数:
    img (str): 图像文件的路径。
    返回:
    str: 生成的描述文本。
    """

    # 根据设备可用性选择CUDA或CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 初始化Blip2处理器和模型，使用预训练的权重
    processor = Blip2Processor.from_pretrained("/HDD1/blip2-opt-2.7b/")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "/HDD1/blip2-opt-2.7b/", torch_dtype=torch.float16
    )
    # 将模型移动到选择的设备
    model.to(device)

    # 打开图像文件
    image = Image.open(img)
    # 对图像进行预处理，准备输入模型
    inputs = processor(images=image, return_tensors="pt", ).to(device, torch.float16)
    # 使用模型生成描述文本
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    # 解码生成的ID为文本
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text

def get_img_pth(directory):
    def extract_numbers_from_path(path):
        return list(map(int, re.findall(r'\d+', path)))

    all_paths = glob.glob(os.path.join(directory, '**', '*'), recursive=True)
    files = [path for path in all_paths if os.path.isfile(path)]
    imgfiles_sorted = sorted(files, key=extract_numbers_from_path) # sort by number

    return imgfiles_sorted



def main():
    """
    '/data/zkf/ModelWeights/Things_dataset/Things_eeg/image_set/img_description/texts_BLIP2_train.npy'
    """
    # 0: check.npy file
    # data = np.load('/data/zkf/ModelWeights/Things_dataset/Things_eeg/image_set/img_description/texts_BLIP2_train.npy')
    # 1: get img pth
    img_files = get_img_pth("/data/zkf/ModelWeights/Things_dataset/Things_eeg/image_set/training_images/")

    # 2: generate text
    texts = []
    for img_pth in img_files:
        text = get_description(img_pth)
        texts.append(text)
        print('hold')

    # 3: arrange text to .npy file
    output_file = '/data/zkf/test.npy'
    np.save(output_file, texts)
    print(f'Texts saved to {output_file}')


if __name__ == '__main__':
    main()
