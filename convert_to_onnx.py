import numpy as np
from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os

from config import get_opts, get_training_size

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3

import datasets.custom_transforms as custom_transforms

from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    elif hparams.model_version == 'v3':
        system = SC_DepthV3(hparams)

    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()

    onnx_input_L = torch.rand(1, 3, 256, 832)
    onnx_input_L = onnx_input_L.to("cuda:0")

    torch.onnx.export(model,
                     onnx_input_L,
                      "{}.onnx".format("SC-DepthV3"),
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['left'],  # the model's input names
                      output_names=['output'])

if __name__ == '__main__':
    main()
