/opt/rocm/bin/hipexamine-perl.sh
sleep 1
perl /opt/rocm/bin/hipify-perl /home/alic-li/RWKV-LM/RWKV-v5-ROCm/cuda/wkv6_cuda.cu > /home/alic-li/RWKV-LM/RWKV-v5-ROCm/hip/wkv6_cuda.cu.hip
