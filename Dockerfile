# Xynz 3D Studio — servidor de IA 3D (TRELLIS.2 da Microsoft, licença MIT), já instalado.
# A máquina da RunPod só liga e gera: nada é compilado na hora. CUDA 12.8 = funciona também nas placas novas (Blackwell).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
    ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/hf PYTHONPATH=/app/TRELLIS.2
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3-pip git ninja-build libgl1 libglib2.0-0 libjpeg-dev libeigen3-dev \
 && ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen && ln -sf /usr/include/eigen3/unsupported /usr/include/unsupported && rm -rf /var/lib/apt/lists/*
RUN python -m pip install -U pip wheel setuptools \
 && pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install imageio tqdm easydict opencv-python-headless trimesh transformers==4.56.2 huggingface_hub \
      zstandard kornia timm pillow runpod requests scipy einops safetensors \
 && pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
RUN git clone -b main --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2 \
 && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && pip install /tmp/CuMesh --no-build-isolation \
 && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && pip install /tmp/FlexGEMM --no-build-isolation \
 && cp -r /app/TRELLIS.2/o-voxel /tmp/o-voxel && CPLUS_INCLUDE_PATH=/usr/include/eigen3 CPATH=/usr/include/eigen3 pip install /tmp/o-voxel --no-build-isolation \
 && rm -rf /tmp/CuMesh /tmp/FlexGEMM /tmp/o-voxel
# confere que tudo importa (sem GPU) antes de publicar
RUN python -c "import torch, xformers, runpod, trimesh; print('ok', torch.__version__)" && pip show cumesh flex_gemm o_voxel | grep -E '^(Name|Version)'
COPY diag.py /app/diag.py
RUN cd /app/TRELLIS.2 && python /app/diag.py 2>&1 | tail -60; true
COPY handler.py /app/handler.py
WORKDIR /app/TRELLIS.2
CMD ["python", "-u", "/app/handler.py"]
