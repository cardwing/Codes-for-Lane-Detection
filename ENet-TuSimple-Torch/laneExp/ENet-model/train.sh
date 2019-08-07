#!/usr/bin/env sh
exp=ENet-model
#cd /mnt/lustre/panxingang/videoSeg/
data=/home/houyuenan/ToHou/videoSeg_tohou
laneExp=laneExp
part=bj11test
rm gen/laneE.t7
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 --partition=$part --job-name=${exp} --kill-on-bad-exit=1 \
CUDA_VISIBLE_DEVICES="4,5,6" th main.lua \
   -data  ${data} \
   -train ${data}/list6/list6_train.txt \
   -val ${data}/list6/list6_val.txt \
   -dataset laneE \
   -save ${laneExp}/${exp}/ENet \
   -retrain ${laneExp}/${exp}/ENet_init.t7 \
   -shareGradInput true \
   -nThreads 1 \
   -nGPU 3 \
   -batchSize 12 \
   -maxIter 50000 \
   -LR 0.01 \
   -labelType segExist \
   -backWeight 0.4 \
   -nEpochs 100 \
2>&1|tee ${laneExp}/${exp}/train_finetune_ENet_test.log
