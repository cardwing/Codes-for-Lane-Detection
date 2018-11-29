Tensorflow implementation of ["Spatial As Deep: Spatial CNN for Traffic Scene Understanding"](https://arxiv.org/abs/1712.06080). (SCNN-Tensorflow) 

# To-Do List

- Test SCNN-Tensorflow in different lane detection benchmarks!

# Content

* [Prerequisites](#Prerequisites)
* [SCNN-Tensorflow](#SCNN-Tensorflow)
* [Datasets](#Datasets)
  * [TuSimple](#TuSimple)
  * [CULane](#CULane)
  * [BDD100K](#BDD100K)
* [Performance](#Performance)
* [Others](#Others)
  * [Citation](#Citation)
  * [Acknowledgement](#Acknowledgement)
  * [Contact](#Contact)

# Prerequisites
- [Tensorflow](https://www.tensorflow.org/)

# SCNN-Tensorflow

Please go to [SCNN-Tensorflow](./SCNN-Tensorflow) to see detailed instructions.

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
|Normal|90.6|88.5|
|Crowded|69.7|66.8|
|Night|66.1|62.5|
|No line|43.4|40.6|
|Shadow|66.9|60.4|
|Arrow|84.1|81.4|
|Dazzle light|58.5|53.0|
|Curve|64.4|61.3|
|Crossroad|1990|2043|
|Total|71.6|68.8|

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
