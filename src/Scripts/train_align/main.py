"""
all的用法：在推理的时候指定all，则会返回所有模态的输出
"""
import os
import re
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from src.Scripts.train_align import diffusion_prior
from src.data.components.THINGSEEG_utils import EEGDataset
from src.models.components.Cogcap.Cogcap import Cogcap, Proj_img, Proj_text, Proj_depth
from src.models.components.Cogcap.Cogcap_eval import Top_K_Accuracy
from src.models.components.utils import load_model

project_root = '/data/zkf/EEGCODE/EEG-DECODE-RELEASE/'  # change your root here


class Evaler:
    """
    集成了加载模型，dataset，评估的class
    """

    def __init__(self, modality, device, train=False, batch_size=512,
                 data_path=None, sub=None, ckpt_path=None, unet_ckpt_path=None):
        """
        modality 决定了要加载的模型
        unet_ckpt_path 传入后，就会使得self.unet的权重加载
        ckpt_path 加载的是eegmodel的权重
        """
        self.modality = modality
        self.device = device
        self.subject = sub
        self.ckpt_path = ckpt_path

        # load relevant params
        with open(os.path.join(project_root, 'configs/experiment/brainencoder_all.yaml'), 'r') as file:
            config = yaml.safe_load(file)

        # load model
        self._load_prior_encoder()
        self.modality_model = self._load_modality_model(modality)

        # load model ckpt
        self._model_load(ckpt_path, unet_ckpt_path)

        # load dataset
        self._load_dataset(data_path, sub, train, batch_size, config)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).requires_grad_(False)
        self.top200_acc_test = Top_K_Accuracy(logit_scale=self.logit_scale, k=200, func='calculate_allmodality')

    def _load_prior_encoder(self):
        if self.modality == "all":
            self.eeg_model = [Cogcap(num_subjects=10, num_latents=1024, sequence_length=250).eval().to(self.device),
                              Cogcap(num_subjects=10, num_latents=1024, sequence_length=250).eval().to(self.device),
                              Cogcap(num_subjects=10, num_latents=1024, sequence_length=250).eval().to(self.device)]
            self.u_net = [diffusion_prior.DiffusionPriorUNet(cond_dim=1024, dropout=0.1),
                          diffusion_prior.DiffusionPriorUNet(cond_dim=1024, dropout=0.1),
                          diffusion_prior.DiffusionPriorUNet(cond_dim=1024, dropout=0.1)]
            self.pipe = [diffusion_prior.Pipe(self.u_net[0], device=self.device, modality="image"),
                         diffusion_prior.Pipe(self.u_net[1], device=self.device, modality="text"),
                         diffusion_prior.Pipe(self.u_net[2], device=self.device, modality="depth")]

        else:
            self.eeg_model = Cogcap(num_subjects=10, num_latents=1024, sequence_length=250).eval().to(self.device)
            self.u_net = diffusion_prior.DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
            self.pipe = diffusion_prior.Pipe(self.u_net, device=self.device, modality=self.modality)

    def _load_modality_model(self, mode):
        if mode == 'image':
            return Proj_img().eval().to(self.device)
        elif mode == 'depth':
            return Proj_depth().eval().to(self.device)
        elif mode == 'text':
            return Proj_text().eval().to(self.device)
        elif mode == 'all':
            return [Proj_img().eval().to(self.device), Proj_text().eval().to(self.device),
                    Proj_depth().eval().to(self.device)]
        else:
            raise ValueError(f"wanted mode: image / depth / text, your mode: {mode}")

    def _load_dataset(self, data_path, sub, train, batch_size, config):
        self.train_dataset = EEGDataset(EEGdata_path=data_path,
                                        imagedata_path=config['data']['image_datapath'],
                                        feature_path=config['data']['feature_path'],
                                        subjects=(sub,),
                                        exclude_subject=None,
                                        use_ori_feature=False,
                                        train=train,
                                        classes=None)
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False if train is False else True,
                                      num_workers=0)

        # train 为 true 的时候新增dataloadertest来评估训练unet的情况
        if train is True:
            self.test_dataset = EEGDataset(EEGdata_path=data_path,
                                           imagedata_path=config['data']['image_datapath'],
                                           feature_path=config['data']['feature_path'],
                                           subjects=(sub,),
                                           exclude_subject=None,
                                           use_ori_feature=False,
                                           train=False,
                                           classes=None)
            self.data_loader_test = DataLoader(self.test_dataset,
                                               batch_size=200,
                                               shuffle=False,
                                               num_workers=0)

    def _model_load(self, ckpt_path, unet_ckpt_path=None):
        """
        mode:决定加载什么模型 img / depth / text
        CAUTION: unet_ckpt_path 需要按照 image text depth的顺序填入
        """
        if ckpt_path is not None:
            model_name = []
            if self.modality == 'image':
                model_name.append('EEGmodel_img')
                model_name.append('imagemodel')
            elif self.modality == 'depth':
                model_name.append('EEGmodel_depth')
                model_name.append('depthmodel')
            elif self.modality == 'text':
                model_name.append('EEGmodel_text')
                model_name.append('textmodel')
            elif self.modality == 'all':
                # 顺序是img text depth
                model_name = [['EEGmodel_img', 'EEGmodel_text', 'EEGmodel_depth'],
                              ['imagemodel', 'textmodel', 'depthmodel']]
            else:
                raise ValueError(f"wanted mode: image / depth / text, your mode: {self.modality}")

            for idx, content in enumerate([self.eeg_model, self.modality_model]):
                if self.modality == "all":
                    # all 的时候在遍历一次list
                    # content: [models]
                    for j, model in enumerate(content):
                        load_model(model=model,
                                   model_name=model_name[idx][j],
                                   ckpt_pth=ckpt_path,
                                   map_location=self.device)
                else:
                    load_model(model=content,
                               model_name=model_name[idx],
                               ckpt_pth=ckpt_path,
                               map_location=self.device)

        if unet_ckpt_path is not None:
            if self.modality == "all":
                for i, unet in enumerate(self.u_net):
                    unet.load_state_dict(torch.load(unet_ckpt_path[i], map_location="cuda"))
            else:
                self.u_net.load_state_dict(torch.load(unet_ckpt_path, map_location="cuda"))

    def _select_modality_features(self, img_features, depth_features, text_features):
        if self.modality == 'image':
            return self.modality_model(img_features)
        elif self.modality == 'depth':
            return self.modality_model(depth_features)
        elif self.modality == 'text':
            return self.modality_model(text_features)
        else:
            raise ValueError(f"wanted mode: image / depth / text, your mode: {self.modality}")

    def eval(self):
        """
        返回loss acc
        """
        with torch.no_grad():
            for batch_idx, (eeg_data, label, text, text_features, img, img_features, depth_features, aug_img_features,
                            img_index, index) in enumerate(self.data_loader_test):
                eeg_data = eeg_data.to(self.device)
                # print("eeg_data", eeg_data.shape)
                text_features = text_features.to(self.device)
                label = label.to(self.device)
                img_features = img_features.to(self.device)
                eeg_features = self.eeg_model(eeg_data)
                modality_features = self._select_modality_features(img_features,
                                                                   depth_features,
                                                                   text_features)
                return self.top200_acc_test.calculate_single_modality(EEG_features=eeg_features,
                                                                      targets=label,
                                                                      modality_features=modality_features
                                                                      )

    def train_unet(self, epochs):
        """
        正式训练unet的入口
        """
        ### 创建存储目录 ###
        save_path = os.path.dirname(self.ckpt_path)
        save_path = os.path.dirname(save_path)
        save_path = os.path.join(os.path.dirname(save_path), 'unet_weights_new', self.modality)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        ### 创建存储目录 ###
        self.pipe.train(self.data_loader,
                        save_path=save_path,
                        num_epochs=epochs,
                        learning_rate=1e-3,
                        eeg_model=self.eeg_model,
                        modality_model=self.modality_model,
                        subject=self.subject,
                        testdataloader=self.data_loader_test)

        # save model after training
        torch.save(self.u_net.state_dict(), f'{save_path}/last.pth')

    def get_img_features(self, num_inference_steps=None, guidance_scale=None, debug_mode=True,
                         generator=None):
        """
        利用这类里面的数据，先做一批embedding出来
        """
        with torch.no_grad():
            for batch_idx, (eeg_data, label, text, text_features, img, img_features, depth_features, aug_img_features,
                            img_index, index) in enumerate(self.data_loader):
                eeg_data = eeg_data.to(self.device)
                # print("eeg_data", eeg_data.shape)
                text_features = text_features.to(self.device)
                img_features = img_features.to(self.device)
                depth_features = depth_features.to(self.device)

                if self.modality == "all":
                    eeg_features = [self.eeg_model[0](eeg_data),
                                    self.eeg_model[1](eeg_data),
                                    self.eeg_model[2](eeg_data)]
                else:
                    eeg_features = self.eeg_model(eeg_data)
                    modality_features = self._select_modality_features(img_features,
                                                                       depth_features,
                                                                       text_features)

                if self.modality == "all":
                    generated_embeddings = []
                    for index_unet in range(3):
                        generated_embedding = []
                        for index in range(eeg_features[0].shape[0]):
                            generated_embedding.append(
                                self.pipe[index_unet].generate(c_embeds=eeg_features[index_unet][index:index + 1],
                                                               num_inference_steps=num_inference_steps,
                                                               guidance_scale=guidance_scale,
                                                               generator=generator))
                            # if demo mode, only generate 2 embeddings
                            if debug_mode is True and index == 5:
                                break
                        generated_embedding = torch.stack(generated_embedding, dim=0).squeeze(1)
                        generated_embeddings.append(generated_embedding)

                    # list里面排序顺序是img, text, depth
                    return generated_embeddings, img_features, text_features, depth_features, img
                else:
                    generated_embedding = []
                    for index in range(eeg_features.shape[0]):
                        generated_embedding.append(self.pipe.generate(c_embeds=eeg_features[index:index + 1],
                                                                      num_inference_steps=num_inference_steps,
                                                                      guidance_scale=guidance_scale))
                        # if demo mode, only generate 2 embeddings
                        if debug_mode is True and index == 1:
                            break
                    generated_embedding = torch.stack(generated_embedding, dim=0).squeeze(1)
                    return generated_embedding, depth_features, img_features, img


class ExperimentSelecter:
    """
    copied from top_k_analysis.py
    """

    def __init__(self,
                 base_path=None,
                 select_param=None,
                 param_log=None,
                 range_way=None):
        """
        base_path: 项目所在的路径
        select_param: 选定的会被记录的指标，是dict，包含dict中所有相关的才会读取
        e.g.: {batchsize: 1024} 那就只会记录batchsize 为1024的相关实验
        param_log: 读取对应的hparams.yaml中的param_log的值

        目的：将experiment对应的最佳ckpt的path拿到
        1：将实验结果进行排序
        2：按照顺序找ckpt，找到了这个模态就停止？
        """
        self.base_path = base_path
        self.select_param = select_param
        self.param_log = param_log
        self.range_way = range_way

    def _file_find_by_name(self, start_path, folder_name):
        """
        找到对应folder_name的文件夹并return
        """
        # 遍历指定路径及其子目录
        for root, dirs, files in os.walk(start_path):
            # 检查dirs列表中的每个目录名
            if folder_name in dirs:
                # 如果找到了名为'9'的目录，打印其完整路径
                folder_path = os.path.join(root, folder_name)
                return folder_path

    def _find_best_ckpt_path(self, files):

        pattern = re.compile(r'model_\d+_(\d+\.\d+).pth')
        # 过滤出符合模式的文件，并提取小数部分
        files_with_decimals = [(file, float(pattern.match(file).group(1))) for file in files if pattern.match(file)]
        # 根据小数部分排序文件
        files_with_decimals.sort(key=lambda x: x[1], reverse=True)
        # 返回小数部分最大的文件名
        if files_with_decimals:
            return files_with_decimals[0][0]
        else:
            return ValueError("无最大文件")

    def main(self, output_unet=False):
        """
        最终输出pair dict
        key: yaml文件
        value: csv文件
        """

        def extract_number(path):
            # 使用正则表达式匹配路径中的数字
            match = re.search(r'/(\d+)/checkpoints/', path)
            if match:
                return int(match.group(1))
            return -1  # 如果没有找到数字，返回-1

        output = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file == "train.log":
                    # 这里找到了需要的.log
                    # data = pd.read_csv(os.path.join(root, file), sep=' ', header=None)
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as cont:
                        for line in cont:
                            # 处理每一行
                            if 'ckpt path' in line:
                                output.append(self._extract_ckpt(line))

        output = sorted(output, key=extract_number)  # 按数字排序

        if output_unet is False:
            return output

        ### find bestunet ckpt ###
        unet_output = []  # 要求输出 10个list 每个list里面img, text, depth的顺序
        for i in range(10):
            unet_output_subject = []
            subject_folder = self._file_find_by_name(start_path=self.base_path, folder_name=str(i))
            subject_folder = os.path.join(subject_folder, "unet_weights_new")

            for modality_index in ['image', 'text', 'depth']:
                subject_folder_unet = os.path.join(subject_folder, modality_index)

                if not os.path.exists(subject_folder_unet):
                    continue

                files = os.listdir(subject_folder_unet)

                if 'model_best.pth' in files:
                    unet_ckpt = 'model_best.pth'
                else:
                    unet_ckpt = self._find_best_ckpt_path(files)

                unet_output_subject.append(os.path.join(subject_folder_unet, unet_ckpt))

            unet_output.append(unet_output_subject)

        return output, unet_output

    def _extract_ckpt(self, log_text):
        # 使用正则表达式来匹配路径
        pattern = r"Best ckpt path: (.*)"
        match = re.search(pattern, log_text)

        if match:
            # 提取路径
            ckpt_path = match.group(1)
            return ckpt_path
        else:
            raise ValueError("No match found.")


def select_sub(ckpt_path):
    """
    根据ckpt_path 返回其对应的subject
    """

    def is_single_digit(s):
        """
        不够robust,在面对两位数的时候会有bug
        """
        try:
            # 尝试将字符串转换为浮点数
            float(s)
            # 检查字符串长度是否为1，同时允许负号
            return len(s) == 1 or (len(s) == 2 and s[0] == '-' and s[1].isdigit())
        except ValueError:
            # 如果转换失败，返回 False
            return False

    output = []
    for single_path in ckpt_path:
        while True:
            basename = os.path.basename(single_path)
            single_path = os.path.dirname(single_path)
            if is_single_digit(basename):
                output.append(f"sub-{int(basename) + 1:02d}")
                break

    return output


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # change your experiment pth here
    base_path = "/data/zkf/ModelWeights/Things_dataset/LightningHydra/logs/train/seriousmultiruns/2024-11-04_16-52-01/"

    save_pth = 'todo'
    select_param = {"batch_size": 1024}
    selecter = ExperimentSelecter(base_path=base_path, select_param=select_param)
    ckpt_path = selecter.main()
    sub = select_sub(ckpt_path)

    modality = ["image", "text", "depth"]
    # for idx in range(10):
    for content in modality:
        evaler = Evaler(modality=content, device=device, train=True, batch_size=512,
                        data_path="/data/zkf/ModelWeights/Things_dataset/Things_eeg/preprocessed_data_250hz/",
                        sub=sub[7], ckpt_path=ckpt_path[7], unet_ckpt_path=None)
        evaler.train_unet(epochs=300)
