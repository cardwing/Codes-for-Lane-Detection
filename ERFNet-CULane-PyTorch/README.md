
### Requirements
- [PyTorch 0.3.0](https://pytorch.org/get-started/previous-versions/).
- Matlab (for tools/prob2lines), version R2014a or later.
- Opencv (for tools/lane_evaluation), version 2.4.8 (later 2.4.x should also work).

### Before Start

Please follow [list](./list) to put CULane in the desired folder. We'll call the directory that you cloned ERFNet-CULane-PyTorch as `$ERFNet_ROOT`.

### Testing
1. Download our trained models to `./trained`
    ```Shell
    cd $ERFNet_ROOT/trained
    ```
   The trained model has already been there.

2. Run test script
    ```Shell
    cd $ERFNet_ROOT
    sh ./test_erfnet.sh
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
    cd $ERFNet_ROOT/tools/lane_evaluation
    make
    sh Run.sh   # it may take over 30min to evaluate
    ```
    Note: `Run.sh` evaluate each scenario separately while `run.sh` evaluate the whole. You may use `calTotal.m` to calculate overall performance from all senarios.  
    By now, you should be able to reproduce the result (F1-measure: 73.1).
    
### Training
1. Download the pre-trained model
    ```Shell
    cd $ERFNet_ROOT/pretrained
    ```
   The pre-trained model has already been there.
2. Training ERFNet model
    ```Shell
    cd $ERFNet_ROOT
    sh ./train_erfnet.sh
    ```
    The training process should start and trained models would be saved in `trained` by default.  
    Then you can test the trained model following the Testing steps above. If your model position or name is changed, remember to set them to yours accordingly.

