import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from src.data.components.utils.depthanything.dpt import DepthAnything
from src.data.components.utils.depthanything.transform import Resize, NormalizeImage, PrepareForNet


def get_model():
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    encoder = 'vitb'  # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder]).eval().to(DEVICE)
    depth_anything.load_state_dict(
        torch.load("/HDD2/Things_dataset/model_pretrained/DepthAnythingWeights/depth_anything_vitb14.pth"))

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    return depth_anything, DEVICE

def img_depth():
    saved_path = "/HDD2/Things_dataset/Things_eeg/image_set/img_path/"
    img_embedding = {}
    model, DEVICE = get_model()
    with torch.no_grad():
        # , 'test_image.npy'
        for i, file_name in enumerate(['train_image.npy', 'test_image.npy']):
            file_name = os.path.join(saved_path, file_name)
            path = np.load(file_name).tolist()
            for j, img_path in enumerate(path):
                ### img_path getted ###
                print(f"{j} / {len(path)}")
                get_depth(img_path, model, DEVICE)

def get_depth(img_path, depth_anything, DEVICE):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str,
                        default="/HDD2/Things_dataset/Things_eeg/image_set/training_images/00002_abacus/abacus_07s.jpg")
    parser.add_argument('--outdir', type=str, default='/HDD2/Things_dataset/vis_depth')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()
    args.img_path = img_path
    args.pred_only = True
    args.outdir = os.path.dirname(args.img_path.replace("image_set", "image_depth_set"))

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            ### depth's shape: (500, 500)
            depth = depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        filename = os.path.basename(filename)
        cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '.png' ), depth)



if __name__ == '__main__':
    img_depth()