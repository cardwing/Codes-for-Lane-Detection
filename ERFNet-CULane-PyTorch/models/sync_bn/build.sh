echo "Compiling sync_bn_kernel kernels..."
if [ -f src/cuda/sync_bn_kernel.o ]; then
    rm src/cuda/sync_bn_kernel.o
fi
if [ -d _ext ]; then
    rm -rf _ext
fi

cd src/cuda
nvcc -c -o sync_bn_kernel.o sync_bn_kernel.cu \
     -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python3 build.py
