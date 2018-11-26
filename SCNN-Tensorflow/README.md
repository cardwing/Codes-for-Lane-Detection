Tensorflow version of SCNN in CULane.

## Installation
    conda create -n tensorflow_gpu pip python=3.5
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.3.0
    pip3 install -r lane-detection-model/requirements.txt 

## Download VGG-16 
Download the vgg.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in lane-detection-model/data.

## Pre-trained model for testing
Download the pre-trained model here. (Coming soon!)

## Test
1. Get probability maps
    ```cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path path/to/model_weights_file --image_path path/to/image_name_list```

Note that path/to/image_name_list should be like [test_img.txt](./lane-detection-model/demo_file/test_img.txt).

    Testing results (probability map of lane markings) are saved in `experiments/predicts/` by default.

2. Get curve line from probability map
    ```Shell
    cd tools/prob2lines
    matlab -nodisplay -r "main;exit"  # or you may simply run main.m from matlab interface
    ```
    The generated line coordinates would be saved in `tools/prob2lines/output/` by default.

3. Calculate precision, recall, and F-measure
    ```Shell
    cd $SCNN_ROOT/tools/lane_evaluation
    make
    sh Run.sh   # it may take over 30min to evaluate
    ```
    Note: `Run.sh` evaluate each scenario separately while `run.sh` evaluate the whole. You may use `calTotal.m` to calculate overall performance from all senarios.

## Train
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/

Note that path/to/CULane-dataset/ should contain files like [train_gt.txt](./lane-detection-model/demo_file/train_gt.txt) and [val_gt.txt](./lane-detection-model/demo_file/train_gt.txt).
