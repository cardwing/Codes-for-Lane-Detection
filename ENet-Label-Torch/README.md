
### Requirements
- [Torch](http://torch.ch/docs/getting-started.html), please follow the installation instructions at [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
- Matlab (for tools/prob2lines), version R2014a or later.
- Opencv (for tools/lane_evaluation), version 2.4.8 (later 2.4.x should also work).
- Hardware: 
For testing, GPU with 3G memory suffices.
For training, we recommend 4xGPU with 12G memory.

### Before Start

Please follow [SCNN-Torch](https://github.com/XingangPan/SCNN) to put CULane in the desired folder. We'll call the directory that you cloned ENet-Label-Torch as `$ENet_ROOT`.

### Testing
1. Download our trained models to `./experiments/pretrained`
    ```Shell
    cd $ENet_ROOT/experiments/pretrained
    ```
   Download [the trained model](https://drive.google.com/open?id=1pIMThIsGn8z8rIs6WgSNzom1H8WVvP5Q) here.

2. Run test script
    ```Shell
    cd $ENet_ROOT
    sh ./experiments/test.sh
    ```
    Testing results (probability map of lane markings) are saved in `experiments/predicts/` by default.

3. Get curve line from probability map
    ```Shell
    cd tools/prob2lines
    matlab -nodisplay -r "main;exit"  # or you may simply run main.m from matlab interface
    ```
    The generated line coordinates would be saved in `tools/prob2lines/output/` by default.

4. Calculate precision, recall, and F-measure
    ```Shell
    cd $ENet_ROOT/tools/lane_evaluation
    make
    sh Run.sh   # it may take over 30min to evaluate
    ```
    Note: `Run.sh` evaluate each scenario separately while `run.sh` evaluate the whole. You may use `calTotal.m` to calculate overall performance from all senarios.  
    By now, you should be able to reproduce the result (F1-measure: 72.0).
    
### Training
1. Download the pre-trained model
    ```Shell
    cd $ENet_ROOT/experiments/models
    ```
   Download the pre-trained model [here](https://drive.google.com/open?id=1Niz4tXMcxIacDIVRZo91AD8xkYIQ23rR) and move it to `$ENet_ROOT/experiments/models/vgg_SCNN_DULR_w9`.
2. Training ENet-Label model
    ```Shell
    cd $ENet_ROOT
    sh ./experiments/train.sh
    ```
    The training process should start and trained models would be saved in `experiments/models/vgg_SCNN_DULR_w9` by default.  
    Then you can test the trained model following the Testing steps above. If your model position or name is changed, remember to set them to yours accordingly.

