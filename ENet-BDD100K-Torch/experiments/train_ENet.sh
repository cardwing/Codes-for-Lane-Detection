#!/usr/bin/env sh
data=./data/bdd100k
CUDA_VISIBLE_DEVICES="6,7" th main.lua \
   -data  ${data} \
   -train ${data}/list/train_gt_bdd.txt \
   -val ${data}/list/val_gt_bdd.txt \
   -dataset lane \
   -save experiments/models/ENet-new \
   -retrain experiments/models/ENet-init/ENet-init.t7 \
   -shareGradInput true \
   -nThreads 2 \
   -nGPU 2 \
   -batchSize 4 \
   -maxIter 60000 \
   -LR 0.01 \
   -backWeight 0.4 \
   -nEpochs 100 \
2>&1|tee experiments/models/train.log
