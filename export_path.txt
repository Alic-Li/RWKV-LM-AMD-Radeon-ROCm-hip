export ROCM_PATH=/opt/rocm 
export HSA_OVERRIDE_GFX_VERSION=10.3.0 
sudo usermod -aG render $USERNAME 
sudo usermod -aG video $USERNAME 
