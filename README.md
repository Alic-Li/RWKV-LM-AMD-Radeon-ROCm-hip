# Transplant the RWKV-LM to ROCm platform. No body except red ream.But I never give up!

## Getting Started (linux only)
1. Install Pytorch requirements (https://pytorch.org/get-started/locally/)(suggest ROCm5.7)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```
2.  install pip requirements
```
pip install -r requirements.txt
```
## Install the ROCm

jump to(https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)


## Launch Application
### run thses commands in you terminal before you launch (It maybe have some changes in our deivces)
```
export ROCM_PATH=/opt/rocm 
export HSA_OVERRIDE_GFX_VERSION=10.3.0 
sudo usermod -aG render $USERNAME 
sudo usermod -aG video $USERNAME 
```
## run script

```
python3 webui.py
```
## We need the effor of every ROCm users or developer!
## 欢迎大家来开发和移植
## 作者废物一个，相信大家一起能做得更好！
