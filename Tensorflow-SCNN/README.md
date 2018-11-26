# Tensorflow-SCNN
Tensorflow version of SCNN in CULane. (Final evaluation is on-going!)

### Prerequisites
- [Tensorflow](https://www.tensorflow.org/)

### Installation
    conda create -n tensorflow_gpu pip python=3.5
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.3.0

Download the vgg.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in the lane-detection-model/data.

### Testing
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py 

### Training
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir /path/to/CULane-dataset/


### Acknowledgement
This repo is built upon [SCNN](https://github.com/XingangPan/SCNN) and [LaneNet](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

### Contact
If you have any problems in reproducing the results, just raise an issue in this repo.
