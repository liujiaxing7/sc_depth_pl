import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import os
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets.custom_transforms as custom_transforms
from config import get_opts, get_training_size
from datasets.test_folder import TestSet
from losses.loss_functions import compute_errors
from SC_Depth import SC_Depth
from visualization import *

import cv2
import re

DATA_TYPE = ['kitti', 'kitti15', 'indemind', 'depth', 'i18R']

def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--model_path', type=str, default='state_dicts/kitti2015.pth')
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--output', type=str)
    parser.add_argument('--bf', type=float, default=14.2)

    # model
    parser.add_argument('--model_version', type=str,
                        default='v1', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--resnet_layers', type=int, default=18)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')


    # configure file
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    args = parser.parse_args()

    return args

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        pass

    return file_list

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        paths = Walk(path, ['jpg', 'png', 'jpeg'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    left_files, right_files = [], []
    if 'kitti' == flag:
        left_files = [f for f in paths if 'image_02' in f]
        right_files = [f.replace('/image_02/', '/image_03/') for f in left_files]
    elif 'kitti15' == flag:
        left_files = [f for f in paths if 'image_2' in f]
        right_files = [f.replace('/image_2/', '/image_3/') for f in left_files]
    elif 'indemind' == flag:
        left_files = [f for f in paths if 'cam0' in f]
        right_files = [f.replace('/cam0/', '/cam1/') for f in left_files]
    elif 'depth' == flag:
        left_files = [f for f in paths if 'left' in f]
        right_files = [f.replace('/left/', '/right/') for f in left_files]
    elif 'i18R' == flag:
        left_files = [f for f in paths if '.L' in f]
        right_files = [f.replace('L/', 'R/').replace('L.', 'R.') for f in left_files]
    else:
        raise Exception("Do not support mode: {}".format(flag))

    return left_files, right_files, root_len

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)


    return depth_img_rgb.astype(np.uint8)

def WriteDepth(depth, limg, path, name, bf):
    name = os.path.splitext(name)[0] + ".png"
    output_concat_color = os.path.join(path, "concat_color", name)
    output_concat_gray = os.path.join(path, "concat_gray", name)
    output_gray = os.path.join(path, "gray", name)
    output_depth = os.path.join(path, "depth", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)
    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_gray)
    MkdirSimple(output_depth)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)

    predict_np = depth.squeeze().cpu().numpy()

    depth_img = bf / predict_np * 100  # to cm

    predict_np_int = predict_np.astype(np.uint8)
    color_img = cv2.applyColorMap(predict_np_int, cv2.COLORMAP_HOT)
    limg_cv = limg  # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_temp = bf / predict_np_int * 100  # to cm
    depth_img_rgb = GetDepthImg(depth_img)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_gray, predict_np)
    cv2.imwrite(output_depth, depth_img_rgb)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)

@torch.no_grad()
def main():
    args = GetArgs()

    output_directory = args.output

    if not args.no_cuda:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # hparams = get_opts()

    # initialize network
    system = SC_Depth(args)

    # load ckpts
    system = system.load_from_checkpoint(args.ckpt_path, strict=False)

    model = system.depth_net

    # get training resolution
    training_size = get_training_size('kitti')

    # data loader
    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    for k in DATA_TYPE:
        left_files, right_files, root_len = GetImages(args.data_path, k)

        if len(left_files) != 0:
            break

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if use_cuda:
        model.cuda()
    model.eval()

    # ckpt = torch.load(args.load_path)
    # model.load_state_dict(ckpt['state_dict'])

    mae = 0
    op = 0
    for left_image_file in tqdm(left_files):
        if not os.path.exists(left_image_file):
            continue

        output_name = left_image_file[root_len+1:]
        img_org = cv2.imread(left_image_file).astype(np.float32)
        img, _ = test_transform([img_org], None)
        img = img[0]

        img_tensor = img.unsqueeze(0).cuda()

        with torch.no_grad():
            pred_disp = model(img_tensor)

        img_org = cv2.resize(img_org, (training_size[1], training_size[0]))
        WriteDepth(pred_disp, img_org, args.output, output_name, args.bf)



if __name__ == '__main__':
    main()