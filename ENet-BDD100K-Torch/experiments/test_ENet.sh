#!/usr/bin/env sh
data=./data/bdd100k
CUDA_VISIBLE_DEVICES="6,7" th main.lua \
   -data  ${data} \
   -train ${data}/list/train_gt_bdd.txt \
   -val ${data}/list/test_gt_bdd.txt \
   -dataset lane \
   -save experiments/models/ENet-tmp \
   -retrain experiments/models/ENet-new/ENet-trained.t7 \
   -shareGradInput true \
   -nThreads 2 \
   -nGPU 2 \
   -batchSize 12 \
   -testOnly true \
   -maxIter 60000 \
   -LR 0.01 \
   -backWeight 0.4 \
   -nEpochs 100 \
2>&1|tee experiments/models/test.log
