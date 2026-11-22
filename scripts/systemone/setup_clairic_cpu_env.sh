#!/bin/bash
# CPU-only conda env for Clairic Flash (Cloudflare/clairic-flash custom Transformers runtime).
set -o pipefail
L=/var/tmp/clairic-cpu/setup_env.log
source /home/letsrtfm/miniforge3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda create -y -q -n clairic-cpu python=3.12 >> $L 2>&1 || { echo "conda create failed" >> $L; echo DONE >> $L; exit 1; }
conda activate clairic-cpu
pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" "torchvision==0.28.0" >> $L 2>&1; echo "torch rc=$?" >> $L
pip install -q "transformers==5.10.2" "kernels==0.13.0" "tomlkit==0.13.3" "einops==0.8.1" "xgrammar==0.2.1" "accelerate>=1.13.0" \
  "huggingface_hub>=1.18.0" "safetensors>=0.8.0" "Pillow>=12.2.0" "pydantic>=2.13.4,<3" "jsonschema>=4.26.0,<5" "numpy>=2.2.6" \
  "opencv-python-headless>=4.13.0.92" "pypdfium2>=5.14.0" "fastapi" "uvicorn" >> $L 2>&1; echo "deps rc=$?" >> $L
pip install -q --no-deps "flash-linear-attention==0.6.0" "fla-core==0.6.0" >> $L 2>&1; echo "fla rc=$?" >> $L
python -c "import torch, transformers; print('torch', torch.__version__, 'tx', transformers.__version__, 'bf16 cpu', torch.backends.mkldnn.is_available())" >> $L 2>&1
echo DONE >> $L
