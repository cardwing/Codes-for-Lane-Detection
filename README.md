Tensorflow implementation of ["Spatial As Deep: Spatial CNN for Traffic Scene Understanding"](https://arxiv.org/abs/1712.06080). (SCNN-Tensorflow) 

# News

Multi-GPU training has been supported. Just change BATCH_SIZE and GPU_NUM in global_config.py, and then use `CUDA_VISIBLE_DEVICES="0,1,2,3" python file_name.py`. Thanks @ yujincheng08.

# Content

* [Installation](#Installation)
* [Datasets](#Datasets)
  * [TuSimple](#TuSimple)
  * [CULane](#CULane)
  * [BDD100K](#BDD100K)
* [SCNN-Tensorflow](#SCNN-Tensorflow)
  * [Test](#Test)
  * [Train](#Train)
* [Performance](#Performance)
* [Others](#Others)
  * [Citation](#Citation)
  * [Acknowledgement](#Acknowledgement)
  * [Contact](#Contact)

# Installation

1. Install necessary packages:
```
    conda create -n tensorflow_gpu pip python=3.5
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.3.0
    pip3 install -r SCNN-Tensorflow/lane-detection-model/requirements.txt
```

2. Download VGG-16:

Download the vgg.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in SCNN-Tensorflow/lane-detection-model/data.

3. Pre-trained model for testing:

Download the pre-trained model [here](https://drive.google.com/open?id=1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5).

# Datasets

## TuSimple

The ground-truth labels of TuSimple testing set is now available at [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). The annotated training (#frame = 3268) and validation labels (#frame = 358) can be found [here](https://github.com/cardwing/Codes-for-Lane-Detection/issues/11), please use them (list-name.txt) to replace the train_gt.txt and val_gt.txt in [train_lanenet.py](./SCNN-Tensorflow/lane-detection-model/tools/train_lanenet.py). Moreover, you need to resize the image to 256 x 512 instead of 288 x 800 in TuSimple. Remember to change the maximum index of rows and columns, and detailed explanations can be seen [here](https://github.com/cardwing/Codes-for-Lane-Detection/issues/18). Please evaluate your pred.json using the labels and [this script](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py). Besides, to generate pred.json, you can refer to [this issue](https://github.com/cardwing/Codes-for-Lane-Detection/issues/4).

## CULane

The whole dataset is available at [CULane](https://xingangpan.github.io/projects/CULane.html).

## BDD100K

The whole dataset is available at [BDD100K](http://bdd-data.berkeley.edu/).

# SCNN-Tensorflow

## Test
    cd SCNN-Tensorflow/lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path path/to/model_weights_file --image_path path/to/image_name_list --save_dir to_be_saved_dir

Note that path/to/image_name_list should be like [test_img.txt](./SCNN-Tensorflow/lane-detection-model/demo_file/test_img.txt). Now, you get the probability maps from our model. To get the final performance, you need to follow [SCNN](https://github.com/XingangPan/SCNN) to get curve lines from probability maps as well as calculate precision, recall and F1-measure.

Reminder: you should check [lanenet_data_processor.py](./SCNN-Tensorflow/lane-detection-model/data_provider/lanenet_data_processor.py) and [lanenet_data_processor_test.py](./SCNN-Tensorflow/lane-detection-model/data_provider/lanenet_data_processor.py) to ensure that the processing of image path is right. You are recommended to use the absolute path in your image path list. Besides, this code needs batch size used in training and testing to be consistent. To enable arbitrary batch size in the testing phase, please refer to [this issue](https://github.com/cardwing/Codes-for-Lane-Detection/issues/10).

## Train
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/

Note that path/to/CULane-dataset/ should contain files like [train_gt.txt](./SCNN-Tensorflow/lane-detection-model/demo_file/train_gt.txt) and [val_gt.txt](./SCNN-Tensorflow/lane-detection-model/demo_file/train_gt.txt).

# Performance

1. TuSimple testing set:

|Model|Accuracy|FP|FN|
|:---:|:---:|:---:|:---:|
|[SCNN-Torch](https://github.com/XingangPan/SCNN)|96.53%|0.0617|0.0180|
|SCNN-Tensorflow|--|--|--|

The pre-trained model for testing is here. (coming soon!) Note that in TuSimple, SCNN-Torch is based on ResNet-101 while SCNN-Tensorflow is based on VGG-16. In CULane and BDD100K, both SCNN-Torch and SCNN-Tensorflow are based on VGG-16.

2. CULane testing set (F1-measure):

|Category|[SCNN-Torch](https://github.com/XingangPan/SCNN)|SCNN-Tensorflow|
|:---:|:---:|:---:|
|Normal|90.6|90.2|
|Crowded|69.7|71.9|
|Night|66.1|64.6|
|No line|43.4|45.8|
|Shadow|66.9|73.8|
|Arrow|84.1|83.8|
|Dazzle light|58.5|59.5|
|Curve|64.4|63.4|
|Crossroad|1990|4137|
|Total|71.6|71.3|

The pre-trained model for testing is [here](https://drive.google.com/open?id=1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5). Note that you need to exchange the order of VGG-MEAN in test_lanenet.py and change the order of input images from RGB to BGR since the pre-trained model uses opencv to read images. You can further boost the performance by referring to [this issue](https://github.com/cardwing/Codes-for-Lane-Detection/issues/5).

3. BDD100K testing set:

|Model|Accuracy|IoU|
|:---:|:---:|:---:|
|[SCNN-Torch](https://github.com/XingangPan/SCNN)|35.79%|15.84|
|SCNN-Tensorflow|--|--|

The accuracy and IoU of lane pixels are computed. The pre-trained model for testing is here. (coming soon!)

# Others

## Citation

If you use the codes, please cite the following publications:

``` 
@inproceedings{pan2018SCNN,  
  author = {Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, and Xiaoou Tang},  
  title = {Spatial As Deep: Spatial CNN for Traffic Scene Understanding},  
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},  
  month = {February},  
  year = {2018}  
}
```

## Acknowledgement
This repo is built upon [SCNN](https://github.com/XingangPan/SCNN) and [LaneNet](https://github.com/MaybeShewill-CV/lanenet-lane-detection).

## Contact
If you have any problems in reproducing the results, just raise an issue in this repo.

## To-Do List
- [ ] Test SCNN-Tensorflow in TuSimple and BDD100K
- [ ] Provide detailed instructions to run SCNN-Tensorflow in TuSimple and BDD100K
- [ ] Upload our light-weight model (ENet-SAD) and its training & testing scripts
