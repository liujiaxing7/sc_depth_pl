# SC_Depth:

This repo provides the pytorch lightning implementation of **SC-Depth** (V1, V2, and V3) for **self-supervised learning of monocular depth from video**.

In the SC-DepthV1 ([IJCV 2021](https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf) & [NeurIPS 2019](https://papers.nips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)), we propose (i) **geometry consistency loss** for scale-consistent depth prediction over time and (ii) **self-discovered mask** for detecting and removing dynamic regions and occlusions during training towards higher accuracy. The predicted depth is sufficiently accurate and consistent for use in the ORB-SLAM2 system. The below video showcases the estimated depth in the form of pointcloud (top) and color map (bottom right).

[<img src="https://jwbian.net/wp-content/uploads/2020/06/77CXZX@H37PIWDBX0R7T.png" width="600">](https://www.youtube.com/watch?v=OkfK3wmMnpo)

In the SC-DepthV2 ([TPMAI 2022](https://arxiv.org/abs/2006.02708v2)), we prove that the large relative rotational motions in the hand-held camera captured videos is the main challenge for unsupervised monocular depth estimation in indoor scenes. Based on this findings, we propose auto-recitify network (**ARN**) to handle the large relative rotation between consecutive video frames. It is integrated into SC-DepthV1 and jointly trained with self-supervised losses, greatly boosting the performance.

<img src="https://jwbian.net/wp-content/uploads/2020/06/vis_depth.png" width="600">

In the SC-DepthV3 ([ArXiv 2022](https://arxiv.org/abs/2211.03660)), we propose a robust learning framework for accurate and sharp monocular depth estimation in (highly) dynamic scenes. As the photometric loss, which is the main loss in the self-supervised methods, is not valid in dynamic object regions and occlusion, previous methods show poor accuracy in dynamic scenes and blurred depth prediction at object boundaries. We propose to leverage an external pretrained depth estimation network for generating the single-image depth prior, based on which we propose effective losses to constrain self-supervised depth learning. The evaluation results on six challenging datasets including both static and dynamic scenes demonstrate the efficacy of the proposed method.

Qualatative depth estimation results: DDAD, BONN, TUM, IBIMS-1

<img src="https://jwbian.net/Demo/vis_ddad.jpg" width="400"> <img src="https://jwbian.net/Demo/vis_bonn.jpg" width="400"> <img src="https://jwbian.net/Demo/vis_tum.jpg" width="400"> <img src="https://jwbian.net/Demo/vis_ibims.jpg" width="400">

Demo Videos



https://user-images.githubusercontent.com/11647217/201716221-94fb20ec-0947-4ea0-b83e-572ffa9a46b5.mp4



<img align="left" src="https://user-images.githubusercontent.com/11647217/201711956-7d2c2f48-8d3c-4c05-9402-9e4115e4b5d7.mp4" width="400"> 
<img align="left" src="https://user-images.githubusercontent.com/11647217/201712014-decd56ba-16eb-4772-90fb-200d489c309c.mp4" width="400"> 







## Install
安装 torch 1.7.1 版本
```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html pytorch-lightning==1.5.7
pip install scipy==1.8.1
pip install kornia==0.5.5
pip install -r requirements.txt
```

或者 torch 11.7 版本
```
conda create -n sc_depth_env python=3.8
conda activate sc_depth_env
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning==1.7.3
pip install kornia
pip install -r requirements.txt
```

## Dataset

We organize the video datasets into the following format for training and testing models:

    Dataset
      -Training
        --Scene0000
          ---*.jpg (list of color images)
          ---cam.txt (3x3 camera intrinsic matrix)
          ---depth (a folder containing ground-truth depth maps, optional for validation)
          ---leres_depth (a folder containing psuedo-depth generated by LeReS, it is required for training SC-DepthV3)
        --Scene0001
        ...
        train.txt (containing training scene names)
        val.txt (containing validation scene names)
      -Testing
        --color (containg testing images)
        --depth (containg ground-truth depths)
        --seg_mask (containing semantic segmentation masks for depth evaluation on dynamic/static regions)

We provide pre-processed datasets:

[**[kitti, nyu, ddad, bonn, tum]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUFwH6FrHGCuh_y6?e=RxOheF) 


## Training

We provide a bash script ("scripts/run_train.sh"), which shows how to train on kitti, nyu, and datasets. Generally, you need edit the config file (e.g., "configs/v1/kitti.txt") based on your devices and run
```bash
python train.py --config $CONFIG --dataset_dir $DATASET
```
Then you can start a `tensorboard` session in this folder by running
```bash
tensorboard --logdir=ckpts/
```
By opening [https://localhost:6006](https://localhost:6006) on your browser, you can watch the training progress.  


## Train on Your Own Data

You need re-organize your own video datasets according to the above mentioned format for training. Then, you may meet three problems: (1) no ground-truth depth for validation; (2) hard to choose an appropriate frame rate (FPS) to subsample videos; (3) no pseudo-depth for training V3.

### No GT depth for validation
Add "--val_mode photo" in the training script or the configure file, which uses the photometric loss for validation. 
```bash
python train.py --config $CONFIG --dataset_dir $DATASET --val_mode photo
```

### Subsample video frames (to have sufficient motion) for training 
We provide a script ("generate_valid_frame_index.py"), which computes and saves a "frame_index.txt" in each training scene. It uses the opencv-based optical flow method to compute the camera shift in consecutive frames. You might need to change the parameters for detecting sufficient keypoints in your images if necessary (usually you do not need). Once you prepare your dataset as the above-mentioned format, you can call it by running
```bash
python generate_valid_frame_index.py --dataset_dir $DATASET
```
Then, you can add "--use_frame_index" in the training script or the configure file to train models on the filtered frames.
```bash
python train.py --config $CONFIG --dataset_dir $DATASET --use_frame_index
```

### Generating Pseudo-depth for training V3

We use the [LeReS](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS) to generate pseudo-depth in this project. You need to install it and generate pseudo-depth for your own images (the pseudo-depth for standard datasets have been provided above). More specifically, you can refer to the code in [this line](https://github.com/aim-uofa/AdelaiDepth/blob/803abcfc186b5cda73c5ca4c369f350e44a8ae1b/LeReS/Minist_Test/tools/test_shape.py#L134) for saving the pseudo-depth.

Besides, it is also possible to use other state-of-the-art monocular depth estimation models to generate psuedo-depth, such as [DPT](https://github.com/isl-org/DPT).


## Pretrained models

[**[Models]**](https://1drv.ms/u/s!AiV6XqkxJHE2mULfSmi4yy-_JHSm?e=s97YRM) 

You need uncompress and put it into "ckpts" folder. Then you can run "scripts/run_test.sh" or "scripts/run_inference.sh" with the pretrained model. 

For v1, we provide models trained on KITTI and DDAD.

For v2, we provide models trained on NYUv2.

For v3, we provide models trained on KITTI, NYUv2, DDAD, BONN, and TUM.


## Testing (Evaluation on Full Images)

We provide the script ("scripts/run_test.sh"), which shows how to test on kitti, nyu, and ddad datasets. The script only evaluates depth accuracy on full images. See the next section for an evaluation of depth estimation on dynamic/static regions, separately.

    python test.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT
    

## Demo

A simple demo is given here. You can put your images in "demo/input/" folder and run
```bash
python inference.py --config configs/v3/nyu.txt \
--input_dir demo/input/ \
--output_dir demo/output/ \
--ckpt_path ckpts/nyu_scv3/epoch=93-val_loss=0.1384.ckpt \
--save-vis --save-depth
```
You will see the results saved in "demo/output/" folder.


## Evaluation on dynamic/static regions

You need to use ("scripts/run_inference.sh") firstly to save the predicted depth, and then you can use the ("scripts/run_evaluation.sh") for doing evaluation. A demo on DDAD dataset is provided in these files. Generally, you need do

### Inference
```bash
python inference.py --config $YOUR_CONFIG \
--input_dir $TESTING_IMAGE_FOLDER \
--output_dir $RESULTS_FOLDER \
--ckpt_path $YOUR_CKPT \
--save-vis --save-depth
```

### Evaluation
```bash
python eval_depth.py \
--dataset $DATASET_FOLDER \
--pred_depth=$RESULTS_FOLDER \
--gt_depth=$GT_FOLDER \
--seg_mask=$SEG_MASK_FOLDER
```


## References

#### SC-DepthV1:
**Unsupervised Scale-consistent Depth Learning from Video (IJCV 2021)** \
*Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Zhichao Li, Le Zhang, Chunhua Shen, Ming-Ming Cheng, Ian Reid* 
[**[paper]**](https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf)
```
@article{bian2021ijcv, 
  title={Unsupervised Scale-consistent Depth Learning from Video}, 
  author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Li, Zhichao and Zhang, Le and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian}, 
  journal= {International Journal of Computer Vision (IJCV)}, 
  year={2021} 
}
```
which is an extension of previous conference version:

**Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (NeurIPS 2019)** \
*Jia-Wang Bian, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, Ian Reid* 
[**[paper]**](https://papers.nips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)
```
@inproceedings{bian2019neurips,
  title={Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video},
  author={Bian, Jiawang and Li, Zhichao and Wang, Naiyan and Zhan, Huangying and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

#### SC-DepthV2:
**Auto-Rectify Network for Unsupervised Indoor Depth Estimation (TPAMI 2022)** \
*Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jun Chin, Chunhua Shen, Ian Reid*
[**[paper]**](https://arxiv.org/abs/2006.02708v2)
```
@article{bian2021tpami, 
  title={Auto-Rectify Network for Unsupervised Indoor Depth Estimation}, 
  author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Chin, Tat-Jin and Shen, Chunhua and Reid, Ian}, 
  journal= {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
  year={2021} 
}
```

#### SC-DepthV3:
**SC-DepthV3: Robust Self-supervised Monocular Depth Estimation for Dynamic Scenes (ArXiv 2022)** \
*Libo Sun\*, Jia-Wang Bian\*, Huangying Zhan, Wei Yin, Ian Reid, Chunhua Shen*
[**[paper]**](https://arxiv.org/abs/2211.03660) \
\* denotes equal contribution and joint first author
```
@article{sc_depthv3, 
  title={SC-DepthV3: Robust Self-supervised Monocular Depth Estimation for Dynamic Scenes}, 
  author={Sun, Libo and Bian, Jia-Wang and Zhan, Huangying and Yin, Wei and Reid, Ian and Shen, Chunhua}, 
  journal= {arXiv:2211.03660}, 
  year={2022} 
}
```
