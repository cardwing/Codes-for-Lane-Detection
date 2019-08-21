python3 -u train_erfnet.py CULane ERFNet train_gt val_gt \
                        --lr 0.01 \
                        --gpus 0 1 2 3 \
                        --npb \
                        --resume pretrained/ERFNet_pretrained.tar \
                        -j 12 \
                        -b 12 \
                        --epochs 12 \
                        --img_height 208 \
                        --img_width 976 \
2>&1|tee train_erfnet_culane.log
