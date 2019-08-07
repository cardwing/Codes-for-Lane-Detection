#!/usr/bin/env sh
exp=ENet-model
data=/home/houyuenan/ToHou/videoSeg_tohou
part=bj11test
#cd /mnt/lustre/panxingang/videoSeg/
rm gen/laneE.t7
# MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --partition=$part --job-name=${exp} --kill-on-bad-exit=1 \
CUDA_VISIBLE_DEVICES="7" th testLaneE.lua \
	-model laneExp/ENet-model/ENet/ENet_trained.t7 \
	-data ${data} \
	-train ${data}/list/list_test_new.txt \
	-val ${data}/list/list_test_new.txt \
	-save predicts/ENet_new/${exp} \
	-dataset laneE \
	-shareGradInput true \
	-nThreads 1 \
	-nGPU 1 \
	-batchSize 1 \
	-nLane 6 \
	-labelType seg \
	-smooth true \
2>&1|tee predicts/test_tmp.log
