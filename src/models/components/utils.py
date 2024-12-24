import os

import numpy as np
import torch
import torch.nn.functional as F

from src.models.components.Cogcap.Cogcap import Cogcap


def save_features(img_names, features, name):
    """
    Save features to a file.
    """
    out_feature = []
    for i, text in enumerate(img_names):
        if "depth" in name:
            text = text.replace(".jpg", ".png")
        out_feature.append(features[os.path.basename(text)])

    out_feature = torch.stack(out_feature).squeeze(1)
    torch.save(out_feature,
               f'/HDD2/Things_dataset/model_pretrained/data_features/features_for_eval/{name}')


@torch.no_grad()
def evaluation(model, eeg, text_features, image_features, k, n):
    """
    Evaluate the model on the given features.

    param: n n-way
    param: k top k acc's k ?
    """
    text_features
    num_text = text_features.shape[0]
    text_bs = 10  # 每次测试text_bs个
    text_feats = []
    text_embeds = []
    for i in range(0, num_text, text_bs):
        textfea_input = text_features[i: min(num_text, i + text_bs)]
        text_output = model.text_encoder(textfea_input)
        text_embed = F.normalize(text_output, dim=-1)
        text_embeds.append(text_embed) # after normalization
        text_feats.append(text_output)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    image_feats = []
    image_embeds = []
    for i in range(image_features.shape[0]):
        imagefea_input = image_features[i].unsqueeze(0)
        image_feat = model.visual_encoder(imagefea_input)
        image_embed = F.normalize(image_feat, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    eeg_feats = []
    eeg_embeds = []
    for i in range(eeg.shape[0]):
        eeg_input = eeg[i].unsqueeze(0)
        eeg_feat = model.eeg_encoder(eeg_input)
        eeg_embed = F.normalize(eeg_feat, dim=-1)
        eeg_feats.append(eeg_feat)
        eeg_embeds.append(eeg_embed)

    eeg_feats = torch.cat(eeg_feats, dim=0)
    eeg_embeds = torch.cat(eeg_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((image_features.shape[0], text_features.shape[0]), -100.0).to(eeg.device)

    step = sims_matrix.size(0) // 1
    start = 0
    end = min(sims_matrix.size(0),start+step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        encoder_output = image_feats[start + i].repeat(k, 1)
        output = model.fusion_encoder(img_embed=encoder_output, text_embed=text_feats[topk_idx], eeg_embed=None)
        score = model.itm_head(output)[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score


    # start t2i
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((text_features.shape[0], image_features.shape[0]), -100.0).to(eeg.device)

    step = sims_matrix.size(0)
    start = step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        encoder_output = image_feats[topk_idx]
        output = model.fusion_encoder(img_embed=encoder_output,
                                      text_embed=text_feats[start + i].repeat(k, 1),
                                      eeg_embed=None)
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

def load_model(model=None, model_name=None, ckpt_pth=None, map_location=None):
    from collections import OrderedDict

    if map_location is None:
        raise ValueError('map_location not assigned !')

    new_state_dict = OrderedDict()
    state_dict = torch.load(ckpt_pth, map_location='cuda:0')['state_dict'] # TODO BUG IN TORCH?
    for k, v in state_dict.items():
        if model_name in k:
            name = k[len(model_name) + 1:] # delete name + .
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

if __name__ == '__main__':
    # eeg_input = torch.randn(128, 63, 250)
    # imginput = torch.randn(128, 1024)
    # augimginput = torch.randn(128, 1024)
    # textinput = torch.randn(128, 1024)
    # evaluation(model, eeg_input, textinput, textinput, 128)
    chpt = "/HDD2/zkfWarehouse/LightningHydra/logs/train/runs/2024-06-26_11-54-53/csv/version_0/checkpoints/epoch=22-step=6003.ckpt"
    ckpt2 = "/HDD2/zkfWarehouse/LightningHydra/logs/train/runs/All_modality/csv/version_0/checkpoints/epoch=49-step=13050.ckpt"
    model = Cogcap()

    load_model(model=model,
               model_name='EEGmodel_img',
               ckpt_pth=chpt,
               )


