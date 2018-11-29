Tensorflow implementation of ["Spatial As Deep: Spatial CNN for Traffic Scene Understanding"](https://arxiv.org/abs/1712.06080). (SCNN-Tensorflow) 

# To-Do List

- Test SCNN-Tensorflow in different lane detection benchmarks!

# Content

* [Installation](#Installation)
* [SCNN-Tensorflow](#SCNN-Tensorflow)
  * [Test](#Test)
  * [Train](#Train)
* [Datasets](#Datasets)
  * [TuSimple](#TuSimple)
  * [CULane](#CULane)
  * [BDD100K](#BDD100K)
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

Download the pre-trained model [here](https://drive.google.com/open?id=18jDdLAyqK0wlazkYulAa2RzAIc7r5Gg6).

# SCNN-Tensorflow

## Test
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path path/to/model_weights_file --image_path path/to/image_name_list

Note that path/to/image_name_list should be like [test_img.txt](./lane-detection-model/demo_file/test_img.txt). Now, you get the probability maps from our model. To get the final performance, you need to follow [SCNN](https://github.com/XingangPan/SCNN) to get curve lines from probability maps as well as calculate precision, recall and F1-measure.

## Train
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/

Note that path/to/CULane-dataset/ should contain files like [train_gt.txt](./lane-detection-model/demo_file/train_gt.txt) and [val_gt.txt](./lane-detection-model/demo_file/train_gt.txt).

# Datasets

## TuSimple

The ground-truth labels of TuSimple testing set is now available at [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Please evaluate your pred.json using the labels and [this script](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py). Besides, to generate pred.json, you can refer to [this issue](https://github.com/cardwing/Codes-for-Lane-Detection/issues/4).

## CULane

The whole dataset is available at [CULane](https://xingangpan.github.io/projects/CULane.html).

## BDD100K

The whole dataset is available at [BDD100K](http://bdd-data.berkeley.edu/).

# Performance

1. TuSimple testing set:

|Model|Accuracy|FP|FN|
|:---:|:---:|:---:|:---:|
|[SCNN-Torch](https://arxiv.org/pdf/1712.06080.pdf)|96.53%|0.0617|0.0180|
|SCNN-Tensorflow|--|--|--|

The pre-trained model for testing is here. (coming soon!)

2. CULane testing set (F1-measure):

|Category|[SCNN-Torch](https://arxiv.org/pdf/1712.06080.pdf)|SCNN-Tensorflow|
|:---:|:---:|:---:|
|Normal|90.6|88.7|
|Crowded|69.7|66.7|
|Night|66.1|63.8|
|No line|43.4|41.7|
|Shadow|66.9|61.4|
|Arrow|84.1|81.8|
|Dazzle light|58.5|55.9|
|Curve|64.4|61.6|
|Crossroad|1990|1917|
|Total|71.6|69.3|

The pre-trained model for testing is [here](https://drive.google.com/open?id=18jDdLAyqK0wlazkYulAa2RzAIc7r5Gg6).

3. BDD100K testing set:

|Model|Accuracy|IoU|
|:---:|:---:|:---:|
|[SCNN-Torch](https://arxiv.org/pdf/1712.06080.pdf)|35.79%|15.84|
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
