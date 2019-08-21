python3 -u test_erfnet.py CULane ERFNet train test_img \
                          --lr 0.01 \
                          --gpus 4 5 6 7 \
                          --npb \
                          --resume trained/ERFNet_trained.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -j 10 \
                          -b 5
