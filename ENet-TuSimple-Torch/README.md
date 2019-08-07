### Before Start

Please follow [list6](./list6) and [list](./list) to put the TuSimple dataset (train, val, test) in the desired folder. We'll call the directory that you cloned ENet-TuSimple-Torch as `$ENet_TuSimple_ROOT`. Note that if you use ENet-Label-Torch (i.e., ENet-SAD) as the backbone, you can get around **96.64%** accuracy in TuSimple testing set.

### Testing
1. Run test script
    ```Shell
    cd $ENet_TuSimple_ROOT
    sh ./laneExp/ENet-model/test.sh
    ```
    Testing results (probability map of lane markings) are saved in `predicts/` by default.

2. Generate json file from probability maps
    ```Shell
    python pred_json.py
    ```
    The generated json file would be named `pred_ENet_test.json` by default.

3. Calculate accuracy, FP and FN
    ```Shell
    cd evaluate
    python lane.py pred_ENet_test.json label.json
    ```
    By now, you should be able to reproduce the result (Accuracy: 0.9486756915, FP: 0.0457901133, FN: 0.0386712197).
    
### Training
1. Training ENet model
    ```Shell
    cd $ENet_TuSimple_ROOT
    sh ./laneExp/ENet-model/train.sh
    ```
    The training process should start and trained models would be saved in `laneExp/ENet-model/ENet/` by default.  
    Then you can test the trained model following the Testing steps above. If your model position or name is changed, remember to set them to yours accordingly.

