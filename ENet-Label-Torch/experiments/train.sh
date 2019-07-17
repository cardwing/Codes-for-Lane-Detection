#!/usr/bin/env sh
exp=vgg_SCNN_DULR_w9
data=./data/CULane
CUDA_VISIBLE_DEVICES="0,1,2,3" th main.lua \
   -data  ${data} \
   -train ${data}/list/train_gt.txt \
   -val ${data}/list/val_gt.txt \
   -dataset lane \
   -save experiments/models/${exp} \
   -retrain experiments/models/${exp}/ENet_concat_new.t7 \
   -shareGradInput true \
   -nThreads 8 \
   -nGPU 4 \
   -batchSize 12 \
   -maxIter 100000 \
   -LR 0.01 \
   -backWeight 0.4 \
   -nEpochs 100 \
2>&1|tee experiments/models/${exp}/train.log
